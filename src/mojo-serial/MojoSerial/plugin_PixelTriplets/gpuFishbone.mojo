# Translated from pixeltrack-standalone/src/serial/plugin-PixelTriplets/gpuFishbone.h

from MojoSerial.plugin_PixelTriplets.GPUCACell import GPUCACell


def fishbone(
    hh: GPUCACell.Hits,
    cells: Span[mut=True, GPUCACell, _],
    nCells: UInt32,  # C++ takes uint32_t const*; unused in the body
    # C++ reads the tracks through the cell's own pointer; with the index form
    # the vector has to come in as an argument.
    cellTracks: GPUCACell.CellTracksVector,
    isOuterHitOfCell: Span[GPUCACell.OuterHitOfCell, _],
    nHits: UInt32,
    checkTrack: Bool,
):
    comptime maxCellsPerHit = Int(GPUCACell.maxCellsPerHit)

    var firstY = 0 + 0 * 1
    var firstX: Int = 0

    var x = InlineArray[Float32, maxCellsPerHit](fill=0)
    var y = InlineArray[Float32, maxCellsPerHit](fill=0)
    var z = InlineArray[Float32, maxCellsPerHit](fill=0)
    var n = InlineArray[Float32, maxCellsPerHit](fill=0)

    var d = InlineArray[UInt16, maxCellsPerHit](fill=0)
    var cc = InlineArray[Int32, maxCellsPerHit](fill=0)

    var nt = Int(nHits)
    for idy in range(firstY, nt):
        ref vc = isOuterHitOfCell[idy]
        var s = Int(len(vc))

        if s < 2:
            continue
        ref c0 = cells[Int(vc[0])]
        var xo = c0.get_outer_x(hh)
        var yo = c0.get_outer_y(hh)
        var zo = c0.get_outer_z(hh)
        var sg = 0
        for ic in range(s):
            ref ci = cells[Int(vc[Int32(ic)])]
            if ci.theUsed == 0:
                continue
            if checkTrack and ci.tracks(cellTracks).empty():
                continue
            cc[sg] = Int32(vc[Int32(ic)])
            d[sg] = UInt16(ci.get_inner_detIndex(hh))

            x[sg] = ci.get_inner_x(hh) - xo
            y[sg] = ci.get_inner_y(hh) - yo
            z[sg] = ci.get_inner_z(hh) - zo
            n[sg] = x[sg] * x[sg] + y[sg] * y[sg] + z[sg] * z[sg]
            sg += 1

        if sg < 2:
            continue

        for ic in range(firstX, sg - 1):
            ref ci = cells[Int(cc[ic])]
            for jc in range(ic + 1, sg):
                ref cj = cells[Int(cc[jc])]
                var cos12 = x[ic] * x[jc] + y[ic] * y[jc] + z[ic] * z[jc]
                if d[ic] != d[jc] and cos12 * cos12 >= 0.99999 * n[ic] * n[jc]:
                    if n[ic] > n[jc]:
                        ci.theDoubletId = -1
                        break
                    else:
                        cj.theDoubletId = -1
