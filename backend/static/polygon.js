// Points are original-image coordinates. The first vertex anchors the area calculation.
const cross = (a,b,c) => (b.x-a.x)*(c.y-a.y) - (b.y-a.y)*(c.x-a.x);
const twiceArea = points => points.slice(2).reduce((sum, p, i) => sum + cross(points[0], points[i+1], p), 0);

function intersects(a, b, c, d) {
  for (const axis of ["x", "y"]) {
    if (Math.max(a[axis],b[axis]) < Math.min(c[axis],d[axis]) ||
        Math.max(c[axis],d[axis]) < Math.min(a[axis],b[axis])) return false;
  }
  return cross(a,b,c)*cross(a,b,d) <= 0 && cross(c,d,a)*cross(c,d,b) <= 0;
}

export function simplePolygon(points) {
  const n = points.length;
  if (n < 3 || new Set(points.map(p => [p.x,p.y].join(","))).size !== n) return false;
  for (let i=0; i<n; i++) {
    const prev = points[(i+n-1)%n], a = points[i], b = points[(i+1)%n];
    if (cross(prev,a,b) === 0 && (prev.x-a.x)*(b.x-a.x)+(prev.y-a.y)*(b.y-a.y) >= 0) return false;
    for (let j=i+2; j<n; j++) {
      if (i === 0 && j === n-1) continue;
      if (intersects(a,b,points[j],points[(j+1)%n])) return false;
    }
  }
  return twiceArea(points) !== 0;
}

export function largestQuadrilateral(points) {
  const [a,b,c,d] = [...points].sort((p,q) => p.x-q.x || p.y-q.y);
  // Up to rotation and reversal, four vertices have exactly three possible perimeters.
  let best = null, maxArea = 0;
  for (const ring of [[a,b,c,d], [a,b,d,c], [a,c,b,d]]) {
    if (!simplePolygon(ring)) continue;
    const area = twiceArea(ring);
    if (Math.abs(area) > maxArea) {
      maxArea = Math.abs(area);
      best = area > 0 ? ring : [ring[0],ring[3],ring[2],ring[1]];
    }
  }
  return best;
}
