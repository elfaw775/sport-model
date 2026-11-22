import assert from "node:assert/strict";

// Run from smoke-visits.mjs with an isolated case containing two uploaded films.
export async function checkCaseViewer(page, id) {
  const click = selector => page.locator(selector).click();
  const waitFor = (selector, imageId) => page.waitForFunction((target, value) =>
    document.querySelector(target)?.dataset.imageId === value, {}, selector, imageId);
  const left = '.film-stage[data-pane="left"]';
  const right = '.film-stage[data-pane="right"]';
  const detail = () => page.evaluate(async key => (await fetch("./api/cases/" + key)).json(), id);
  const measured = (imageId, count) => page.waitForFunction(async (key, imageId, count) => {
    const record = await (await fetch("./api/cases/" + key)).json();
    return record.images.find(image => image.id === imageId)?.measurements?.length === count;
  }, {}, id, imageId, count);
  const mark = async (stage, points, drag = false) => {
    const svg = await page.waitForSelector(stage + " svg");
    await svg.scrollIntoView();
    const screenPoints = await page.$eval(stage + " g.image-space", (group, originals) => {
      const matrix = group.getScreenCTM();
      return originals.map(([x, y]) => {
        const point = new DOMPoint(x, y).matrixTransform(matrix);
        return [point.x, point.y];
      });
    }, points);
    for (const [index, [x, y]] of screenPoints.entries()) {
      const onImage = await page.evaluate((x, y, stage) =>
        document.elementFromPoint(x, y)?.closest(stage) != null, x, y, stage);
      assert.ok(onImage, "measurement point must be visible in the chosen pane");
      if (drag) {
        await page.mouse.move(x, y, {steps:index ? 4 : 1});
        if (index === 0) await page.mouse.down();
        else await page.mouse.up();
      } else await page.mouse.click(x, y);
    }
  };
  const checkPoints = (actual, expected, epsilon) => {
    assert.equal(actual.length, expected.length);
    actual.forEach(({x,y}, i) => {
      assert.ok(Math.abs(x - expected[i][0]) <= epsilon && Math.abs(y - expected[i][1]) <= epsilon,
        `source point ${i + 1} (${x}, ${y}) vs ${expected[i].join(", ")}`);
    });
  };
  const scale = () => page.$eval(left + " g.image-space", group => {
    const {a,b,c,d} = group.getScreenCTM();
    return {magnitude: Math.hypot(a,b), orientation: Math.sign(a*d - b*c)};
  });

  await waitFor(left, "primary");
  assert.equal((await detail()).images.length, 2);
  await click('.film-card[data-image-id="1"]');
  await waitFor(left, "1");
  await page.select('#viewer-metadata [name="projection"]', "斜位");
  await page.select('#viewer-metadata [name="quality"]', "一般");
  await click('#viewer-metadata button[type="submit"]');
  await page.waitForFunction(async key => {
    const {images} = await (await fetch("./api/cases/" + key)).json();
    return images[1].projection === "斜位" && images[1].quality === "一般" && images[0].projection === "侧位";
  }, {}, id);
  await click('.film-card[data-image-id="primary"]');
  await waitFor(left, "primary");
  await click("#viewer-compare");
  await waitFor(right, "1");
  assert.notEqual(await page.$eval("#viewer-primary-image", el => el.value),
    await page.$eval("#viewer-compare-image", el => el.value));
  const before = await scale();
  await click('[data-view="zoom-in"]');
  await click('[data-view="flip"]');
  const after = await scale();
  assert.ok(after.magnitude > before.magnitude);
  assert.equal(after.orientation, -before.orientation);

  // For the 220 × 449 primary image these vectors form a 3:4:5 triangle in source pixels.
  const line = [[100,200], [130,240]];
  await click('[data-tool="distance"]');
  await mark(left, line);
  await measured("primary", 1);
  let record = await detail();
  const distance = record.images[0].measurements[0];
  assert.equal(distance.kind, "distance");
  assert.equal(distance.created_by, "朱诗嘉");
  assert.equal(distance.unit, "px");
  assert.ok(Math.abs(distance.value - 50) <= 2, "length should stay 50 source pixels after zoom and flip");
  checkPoints(distance.points, line, 2);
  assert.equal(record.images[1].measurements.length, 0);

  // Measure the second film independently using a right angle on its original 1317 × 1401 grid.
  const anglePoints = [[550,700], [650,700], [650,600]];
  await click('[data-tool="angle"]');
  await mark(right, anglePoints);
  await measured("1", 1);
  record = await detail();
  const angle = record.images[1].measurements[0];
  assert.equal(angle.kind, "angle");
  assert.equal(angle.created_by, "朱诗嘉");
  assert.equal(angle.unit, "°");
  assert.ok(Math.abs(angle.value - 90) <= 3);
  checkPoints(angle.points, anglePoints, 5);
  assert.deepEqual(record.images[0].measurements, [distance]);
  const nameInput = page.locator('#viewer-annotation-form [name=label]');
  await click('[data-tool="polygon"]');
  // Alternate between opposite corners, as during a four-point clinical outline.
  const corners = [[550,650],[650,740],[650,650],[550,740]];
  const previewPoints = shape => page.$eval(right + " .manual-draft " + shape, shape =>
    Array.from(shape.points, p => ({x:p.x,y:p.y})));
  const saveDisabled = () => page.$eval("#save-annotation", button => button.disabled);
  for (let count = 1; count <= 3; count++) {
    await mark(right, [corners[count - 1]]);
    if (count === 1) await nameInput.fill("附片人工多边形");
    checkPoints(await previewPoints("polyline"), corners.slice(0, count), 5);
    assert.equal(await page.$(right + " .manual-draft polygon"), null);
    assert.equal(await saveDisabled(), true, `Cannot submit a ${count}-point outline`);
  }
  assert.equal((await detail()).images[1].annotations.length, 0);
  await mark(right, [[550,700]]);
  const firstBoundary = await previewPoints("polygon");
  assert.equal(firstBoundary.length, 4);
  assert.equal(await saveDisabled(), false);
  await mark(right, [[600,690]]);
  assert.deepEqual(await previewPoints("polygon"), firstBoundary, "Fifth click must leave the four-point boundary intact");
  await click("#undo-vertex");
  checkPoints(await previewPoints("polyline"), corners.slice(0,3), 5);
  assert.equal(await saveDisabled(), true);
  await mark(right, [corners[3]]);
  const draft = await previewPoints("polygon");
  assert.equal(draft.length, 4);
  const area = Math.abs(draft.reduce((sum, p, i) => {
    const next = draft[(i+1) % draft.length];
    return sum + p.x*next.y - next.x*p.y;
  }, 0)) / 2;
  assert.ok(area > 8000 && area < 10000, "Four unordered corners must preview the full region");
  await click("#save-annotation");
  await page.waitForSelector("[data-annotation-id]");
  const polygon = (await detail()).images[1].annotations[0];
  assert.equal(polygon.label, "附片人工多边形");
  checkPoints(polygon.points, draft.map(p => [p.x, p.y]), 0.001);
  page.once("dialog", dialog => dialog.accept());
  await click("[data-delete-annotation]");
  await page.waitForFunction(()=>!document.querySelector("[data-annotation-id]"));
  assert.equal((await detail()).images[1].annotations.length,0);
  // Draw on the flipped primary pane. Coordinates must return to the underlying image grid.
  await click('.film-card[data-image-id="primary"]');
  const activeStage = (await page.$eval(right, el=>el.dataset.imageId)) === "primary" ? right : left;
  await click('[data-tool="rectangle"]');
  await mark(activeStage, [[80,180],[130,230]], true);
  await nameInput.fill("主片可疑区域");
  await click("#save-annotation");
  await page.waitForSelector("[data-annotation-id]");
  const image = (await detail()).images[0], rectangle = image.annotations[0];
  assert.equal(rectangle.kind,"rectangle");assert.equal(rectangle.source,"manual");
  checkPoints(rectangle.points, [[80,180],[130,230]],2);
  await click("[data-rename]");
  await nameInput.fill("主片复核区域");
  await click("#save-annotation");
  await page.waitForFunction(()=>document.querySelector(".annotation-list strong")?.textContent==="主片复核区域");
  await click("#annotation-layer");
  assert.equal(await page.$('[data-manual-id]'),null);
  await click("#annotation-layer");
  await page.waitForSelector('[data-manual-id]');
  const png = await page.$eval("#annotation-export",async link=>{
    const reply=await fetch(link.href); const data=new Uint8Array(await reply.arrayBuffer());
    const header=new DataView(data.buffer);
    return [reply.headers.get("content-type"),data[0],data[1],data[2],data[3],header.getUint32(16),header.getUint32(20)];
  });
  assert.deepEqual(png,["image/png",137,80,78,71,image.width,image.height]);
  await page.reload();
  await page.waitForSelector('[data-manual-id]');
  assert.equal(await page.$eval(".annotation-list strong",el=>el.textContent),"主片复核区域");
  await click('.film-card[data-image-id="1"]');
  await waitFor(left, "1");
  await page.waitForFunction(() => document.querySelector(".measurement-list")?.textContent.includes("°"));
  await page.setViewport({width:390,height:844});
  await click("#viewer-compare");
  await page.waitForSelector(right, {visible:true});
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth),390);
  await page.setViewport({width:1440,height:960});
  return {distance, angle};
}
