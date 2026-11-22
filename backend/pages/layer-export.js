// The original training dataset is a CVAT COCO 1.0 export with one lesion category.
import {renderRegions} from "./review.js";
import {zipSync} from "./vendor/fflate.js";

const categories = [
  {id:1, name:"病变", supercategory:""},
  {id:2, name:"annotation mark", supercategory:""}
];

function marksForLayer(record, image, layer) {
  const marks = [];
  if (layer !== "manual" && image.id === "primary") {
    for (const [index, box] of record.detections.entries()) {
      marks.push({
        reference:"A" + (index + 1), kind:"rectangle", source:"ai",
        points:[{x:box.x1, y:box.y1}, {x:box.x2, y:box.y2}],
        attributes:{"病变类型":"疑似病变", score:box.score, model:record.model}
      });
    }
  }
  if (layer !== "ai") {
    for (const [index, mark] of image.annotations.entries()) {
      marks.push({
        reference:"M" + (index + 1), kind:mark.kind, source:"manual", points:mark.points,
        attributes:{"病变类型":mark.label, annotation_id:mark.id, reviewer:mark.updated_by}
      });
    }
  }
  return marks;
}

function cocoAnnotation(mark, index) {
  const points = mark.points, xs = points.map(p => p.x), ys = points.map(p => p.y);
  const x = Math.min(...xs), y = Math.min(...ys);
  const width = Math.max(...xs) - x, height = Math.max(...ys) - y;
  const polygon = mark.kind === "polygon";
  const area = polygon ? Math.abs(points.reduce((sum, point, i) => {
    const next = points[(i + 1) % points.length];
    return sum + point.x * next.y - next.x * point.y;
  }, 0)) / 2 : width * height;
  return {
    id:index+1, image_id:1, category_id:1, iscrowd:0, area, bbox:[x,y,width,height],
    segmentation:polygon ? [points.flatMap(p => [p.x,p.y])] : [],
    attributes:{source:mark.source, occluded:false, rotation:0, ...mark.attributes}
  };
}

// Images with orientation metadata are exported in display coordinates so that importers
// which ignore EXIF still align the exported vertices with the pixels.
function needsOrientation(bytes, mime) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  function rotatedExif(start, end) {
    if (end > bytes.length || start + 8 > end) return false;
    if (view.getUint32(start) === 0x45786966) start += 6;
    if (start + 8 > end) return false;
    const little = view.getUint16(start) === 0x4949;
    if ((!little && view.getUint16(start) !== 0x4d4d) || view.getUint16(start+2, little) !== 42) return false;
    const ifd = start + view.getUint32(start+4, little);
    if (ifd + 2 > end) return false;
    const count = view.getUint16(ifd, little);
    for (let n=0, offset=ifd+2; n<count && offset+12<=end; n++, offset+=12) {
      if (view.getUint16(offset, little) !== 274 || view.getUint16(offset+2, little) !== 3) continue;
      const orientation = view.getUint16(offset+8, little);
      return orientation >= 2 && orientation <= 8;
    }
    return false;
  }
  if (mime === "image/jpeg") {
    let i = 2;
    while (i + 10 <= bytes.length && bytes[i] === 255) {
      const marker = bytes[++i];
      if (marker === 255) continue;
      if (marker === 0xda || marker === 0xd9) break;
      const length = view.getUint16(i+1);
      if (length < 2) break;
      if (marker === 0xe1 && view.getUint32(i+3) === 0x45786966 && rotatedExif(i+3, i+1+length)) return true;
      i += length + 1;
    }
  } else {
    const png = mime === "image/png";
    let i = png ? 8 : 12;
    while (i + 8 <= bytes.length) {
      const type = view.getUint32(i + (png ? 4 : 0));
      const size = view.getUint32(i + (png ? 0 : 4), !png);
      if (type === (png ? 0x65584966 : 0x45584946) && rotatedExif(i+8, i+8+size)) return true;
      i += png ? 12+size : 8+size+(size%2);
    }
  }
  return false;
}

async function cocoArchive(record, image, marks, layer) {
  let bytes = new Uint8Array(await image.blob.arrayBuffer());
  let ext = {"image/jpeg":"jpg","image/png":"png","image/webp":"webp"}[image.blob.type];
  if (needsOrientation(bytes, image.blob.type)) {
    const oriented = await renderRegions(image, []);
    bytes = new Uint8Array(await oriented.arrayBuffer());
    ext = "png";
  }
  const filename = record.id + "-" + image.id + "." + ext;
  const labels = {
    info:{case_id:record.id, image_id:image.id, original_filename:image.filename, layer},
    licenses:[{id:0,name:"",url:""}], categories,
    images:[{id:1,width:image.width,height:image.height,file_name:filename,license:0, flickr_url:"",coco_url:"",date_captured:0}],
    annotations:marks.map(cocoAnnotation)
  };
  const packed = zipSync({
    ["images/default/" + filename]:bytes,
    "annotations/instances_default.json":new TextEncoder().encode(JSON.stringify(labels))
  }, {level:0});
  return new Blob([packed], {type:"application/zip"});
}

export async function exportLayer(record, imageId, format) {
  const match = /^(combined|ai|manual)\.(png|zip)$/.exec(format);
  const evidence = record.signed_evidence || record;
  const image = evidence.images.find(im => im.id === imageId);
  if (!match || !image) throw Object.assign(new Error("未找到此标注图层。"), {status:404});
  const [, layer, ext] = match;
  const original = record.images.find(im => im.id === imageId);
  const source = {...image, blob:original.blob};
  const marks = marksForLayer(evidence, image, layer);
  const body = ext === "png" ? await renderRegions(source, marks) : await cocoArchive(record, source, marks, layer);
  const disposition = ext === "png" ? "inline" : "attachment";
  const name = record.id + "-" + imageId + "-" + layer + "." + ext;
  return new Response(body, {headers:{"Content-Type":body.type, "Cache-Control":"private, no-store",
    "Content-Disposition":disposition + '; filename="' + name + '"'}});
}
