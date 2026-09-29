import { describe, expect, it } from "vitest";

import { mapView, wallsStatus, type MossMap } from "@/components/SimMossMap";

const base: MossMap = { area: null, rim: [], mem: [], target: null, goal: null, look: null, leg: null, cov: null };
// A 1 m x 0.5 m patch of grid, 0.1 m cells, with two solid cells.
const room = { x0: -1, y0: -1, cell: 0.1, n: 20, solid: [5 * 20 + 3, 12 * 20 + 15], felt: [] as [number, number][], swept: 0.4 };

describe("MOSS's map", () => {
  it("frames the work area once it has one", () => {
    expect(mapView({ ...base, area: [-1.5, 1.5, -1.2, 1.2], room }, [0, 0, 0])).toEqual([-1.5, 1.5, -1.2, 1.2]);
  });
  it("frames what the depth has found, and the robot, before it has an area", () => {
    const v = mapView({ ...base, room }, [0, 0, 0])!;
    expect(v[0]).toBeCloseTo(-1 + 3.5 * 0.1);        // the solid cell at ix 3
    expect(v[1]).toBeCloseTo(-1 + 15.5 * 0.1);       // ...and at ix 15
    expect(v[2]).toBeCloseTo(-0.5);                  // the robot's box reaches lower than the cells
  });
  it("says where the walls came from", () => {
    expect(wallsStatus({ ...base, area: [0, 1, 0, 1], areaSrc: "sensed", room })).toMatch(/depth camera/);
    expect(wallsStatus({ ...base, area: [0, 1, 0, 1], areaSrc: "given" })).toMatch(/GIVEN/);
    expect(wallsStatus({ ...base, room })).toMatch(/40% of a turn/);
  });
});
