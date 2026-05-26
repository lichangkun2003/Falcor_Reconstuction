import falcor

def render_graph_Pass():
    g = RenderGraph("VoxelReconstruction")

    voxelReconstruction_pass = createPass("VoxelReconstruction")
    voxel_pass = createPass("VoxelizationPass_CPU")
    read_pass = createPass("ReadVoxelPass")
    accumulate_pass = createPass("AccumulatePass", {"enabled": True, "precisionMode": "Single",'maxFrameCount': 1024})
    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})


    g.addPass(voxel_pass,"VoxelizationPass")
    g.addPass(read_pass,"ReadVoxelPass")
    g.addPass(voxelReconstruction_pass,"VoxelReconstruction")
    g.addPass(accumulate_pass,"AccumulatePass")

    g.addEdge("VoxelizationPass.dummy","ReadVoxelPass.dummy")

    g.addEdge("ReadVoxelPass.vBuffer","VoxelReconstruction.vBuffer")
    g.addEdge("ReadVoxelPass.gBuffer","VoxelReconstruction.gBuffer")
    g.addEdge("ReadVoxelPass.pBuffer","VoxelReconstruction.pBuffer")
    g.addEdge("ReadVoxelPass.blockMap","VoxelReconstruction.blockMap")



    g.addEdge("VoxelReconstruction.color","AccumulatePass.input")


    
    g.markOutput("VoxelReconstruction.color")
    g.markOutput("AccumulatePass.output")
    return g

Graph = render_graph_Pass()
try: 
    m.addGraph(Graph)
except NameError: 
    pass
