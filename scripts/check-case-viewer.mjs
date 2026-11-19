import assert from "node:assert/strict";

// Run from smoke-visits.mjs with an isolated case containing two uploaded films.
export async function checkCaseViewer(page, id) {
  const click = selector => page.locator(selector).click();
  const waitFor = (selector, imageId) => page.waitForFunction((target, value) =>
    document.querySelector(target)?.dataset.imageId === value, {}, selector, imageId);
  const left = '.film-stage[data-pane="left"]';
  const right = '.film-stage[data-pane="right"]';
  const detail = () => page.evaluate(async key => (await fetch("/api/cases/" + key)).json(), id);
  const measured = (imageId, count) => page.waitForFunction(async (key, imageId, count) => {
    const record = await (await fetch("/api/cases/" + key)).json();
    return record.images.find(image => image.id === imageId)?.measurements?.length === count;
  }, {}, id, imageId, count);
  const mark = async (stage, points) => {
    const svg = await page.waitForSelector(stage + " svg");
    await svg.scrollIntoView();
    const screenPoints = await page.$eval(stage + " g.image-space", (group, originals) => {
      const matrix = group.getScreenCTM();
      return originals.map(([x, y]) => {
        const point = new DOMPoint(x, y).matrixTransform(matrix);
        return [point.x, point.y];
      });
    }, points);
    for (const [x, y] of screenPoints) {
      const onImage = await page.evaluate((x, y, stage) =>
        document.elementFromPoint(x, y)?.closest(stage) != null, x, y, stage);
      assert.ok(onImage, "measurement point must be visible in the chosen pane");
      await page.mouse.click(x, y);
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
    const {images} = await (await fetch("/api/cases/" + key)).json();
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
  await page.reload();
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
