from omni.isaac.utils.scripts.nucleus_utils import find_nucleus_server
import carb
import omni
from pxr import UsdGeom, Gf, Sdf, PhysxSchema, PhysicsSchema, PhysicsSchemaTools
from omni.isaac.synthetic_utils import DomainRandomization




class Environment:
    def __init__(self, omni_kit, z_height=0):
        self.omni_kit = omni_kit

        # 링크 확인
        result, nucleus_server = find_nucleus_server()
        if result is False:
            carb.log_error(
                "Could not find nucleus server with /Isaac folder. Please specify the correct nucleus server in experiences/isaac-sim-python.json"
            )
            return

        # 변수 설정
        self.tile_size = [25.0, 25.0]

        # 1=UP, 2 = DOWN, 3 = LEFT, 4= RIGHT
        self.direction_map = {1: 180, 2: 0, 3: -90, 4: 90}

        self.prims = []  # list of spawned tiles
        self.height = z_height  # height of the ground tiles
        self.tiles = None
        self.state = None
        # because the ground plane is what the robot drives on, we only do this once. We can then re-generate the road as often as we need without impacting physics
        self.setup_physics()
        self.road_map = None
        self.road_path_helper = None

        self.omni_kit.create_prim("/World/Floor", "Xform")

        # cube 생성
        stage = omni.usd.get_context().get_stage()
        cubeGeom = UsdGeom.Cube.Define(stage, "/World/Floor/thefloor")
        cubeGeom.CreateSizeAttr(300)
        offset = Gf.Vec3f(75, 75, -150.1)
        cubeGeom.AddTranslateOp().Set(offset)

        # 환경 load
        # In manual mode, user can control when scene randomization occur whereas in non-manual mode scene randomization is controlled via the duration parameter in various DR components.
        self.dr = DomainRandomization()
        self.dr.toggle_manual_mode()
        self.omni_kit.update(1 / 60.0)
        print("waiting for materials to load...")

        while self.omni_kit.is_loading():
            self.omni_kit.update(1 / 60.0)

        # Create a sphere room so the world is not black
        lights = []
        for i in range(5):
            prim_path = "/World/Lights/light_" + str(i)
            self.omni_kit.create_prim(
                prim_path,
                "SphereLight",
                translation=(0, 0, 200),
                rotation=(0, 0, 0),
                attributes={"radius": 10, "intensity": 1000.0, "color": (1.0, 1.0, 1.0)},
            )
            lights.append(prim_path)

        self.dr.create_light_comp(light_paths=lights)
        self.dr.create_movement_comp(prim_paths=lights, min_range=(0, 0, 30), max_range=(150, 150, 30))

    def generate_lights(self):
        prim_path = omni.kit.utils.get_stage_next_free_path(self.omni_kit.get_stage(), "/World/Env/Light", False)
        self.omni_kit.create_prim(
            prim_path,
            "RectLight",
            translation=(75, 75, 100),
            rotation=(0, 0, 0),
            attributes={"height": 150, "width": 150, "intensity": 2000.0, "color": (1.0, 1.0, 1.0)},
        )

    def reset(self, shape):
        self.generate_warehouse(shape)
        self.dr.randomize_once() # scene을 randomize한다는데 무슨 의미인지 확인해야 함

    def generate_warehouse(self, shape):
        stage = self.omni_kit.get_stage()
        result, nucleus_server = find_nucleus_server()
        if result is False:
            carb.log_error("Could not find nucleus server with /Isaac folder")
            return

        path = nucleus_server + "/Isaac/Environments/Simple_Warehouse/warehouse.usd"
        prefix = "/World/Env/Track"
        prim_path = omni.kit.utils.get_stage_next_free_path(stage, prefix, False)
        track_prim = stage.DefinePrim(prim_path, "Xform")
        track_prim.GetReferences().AddReference(path)

    def setup_physics(self):
        stage = self.omni_kit.get_stage()
        # Add physics scene
        scene = PhysicsSchema.PhysicsScene.Define(stage, Sdf.Path("/World/Env/PhysicsScene"))
        # Set gravity vector
        scene.CreateGravityAttr().Set(Gf.Vec3f(0.0, 0.0, -981.0))
        # Set physics scene to use cpu physics
        PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/World/Env/PhysicsScene"))
        physxSceneAPI = PhysxSchema.PhysxSceneAPI.Get(stage, "/World/Env/PhysicsScene")
        physxSceneAPI.CreatePhysxSceneEnableCCDAttr(True)
        physxSceneAPI.CreatePhysxSceneEnableStabilizationAttr(True)
        physxSceneAPI.CreatePhysxSceneEnableGPUDynamicsAttr(False)
        physxSceneAPI.CreatePhysxSceneBroadphaseTypeAttr("MBP")
        physxSceneAPI.CreatePhysxSceneSolverTypeAttr("TGS")
        # Create physics plane for the ground
        PhysicsSchemaTools.addGroundPlane(
            stage, "/World/Env/GroundPlane", "Z", 100.0, Gf.Vec3f(0, 0, self.height), Gf.Vec3f(1.0)
        )
        # Hide the visual geometry
        imageable = UsdGeom.Imageable(stage.GetPrimAtPath("/World/Env/GroundPlane/geom"))
        if imageable:
            imageable.MakeInvisible()

    # def get_valid_location(self): #### 수정
    #     # keep try until within the center track
    #     dist = 1
    #     x = 4
    #     y = 4
    #     while dist > LANE_WIDTH:
    #         x = random.randint(0, TRACK_DIMS[0])
    #         y = random.randint(0, TRACK_DIMS[1])
    #         dist = center_line_dist(np.array([x, y]))
    #
    #     print("get valid location called", x, y)
    #     return (x, y)
