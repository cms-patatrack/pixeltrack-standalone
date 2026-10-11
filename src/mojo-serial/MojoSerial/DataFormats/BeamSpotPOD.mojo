from MojoSerial.MojoBridge.DTypes import Float, Typeable


@fieldwise_init
struct BeamSpotPOD(Copyable, Defaultable, Movable, Typeable, TrivialRegisterPassable):
    var x: Float  # position
    var y: Float
    var z: Float

    var sigmaZ: Float

    var beamWidthX: Float
    var beamWidthY: Float

    var dxdz: Float
    var dydz: Float

    var emittanceX: Float
    var emittanceY: Float

    var betaStar: Float

    @always_inline
    def __init__(out self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0

        self.sigmaZ = 0.0

        self.beamWidthX = 0.0
        self.beamWidthY = 0.0

        self.dxdz = 0.0
        self.dydz = 0.0

        self.emittanceX = 0.0
        self.emittanceY = 0.0

        self.betaStar = 0.0

    @always_inline
    @staticmethod
    def dtype() -> String:
        return "BeamSpotPOD"
