#include <cstdlib> // random numbers
#include <chrono> // measuring wall time
#include <string>
#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>

inline static float randomFloat() {
  return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
}

poplar::Device getDevice(unsigned int num_ipus) {
  /* return a Poplar device with the desired number of IPUs */
  auto manager = poplar::DeviceManager::createDeviceManager();
  auto devices = manager.getDevices(poplar::TargetType::IPU, num_ipus);
  // Use the first available device
  for (auto &device : devices)
    if (device.attach()) 
      return std::move(device);

  throw std::runtime_error("No hardware device available.");
}

int main (int argc, char** argv) {
  std::size_t num_ipus = 1;
  std::size_t num_iterations = 100000;
  std::size_t size_per_worker = 1024;
  std::string vertex = "TriadVertexMemoryAlignment";

  // Attach to IPU device
  auto device = getDevice(num_ipus);
  auto &target = device.getTarget();
  auto num_tiles = target.getNumTiles();
  auto num_worker_contexts = target.getNumWorkerContexts();

  // Create graph object
  poplar::Graph graph{target};
  graph.addCodelets("codelets.gp");

  // Allocate Tensors, device variables
  auto a = graph.addVariable(poplar::FLOAT, {num_tiles, num_worker_contexts, size_per_worker}, "a");
  auto b = graph.addVariable(poplar::FLOAT, {num_tiles, num_worker_contexts, size_per_worker}, "b");
  auto c = graph.addVariable(poplar::FLOAT, {num_tiles, num_worker_contexts, size_per_worker}, "c");

  // Host variables
  std::size_t total_size = a.numElements();
  std::vector<float> host_a(total_size);
  std::vector<float> host_b(total_size);
  std::vector<float> host_c(total_size);
  float q = randomFloat();
  for (std::size_t i = 0; i < total_size; ++i) {
    host_b[i] = randomFloat();
    host_c[i] = randomFloat();
  }
  
  // Perform tile mapping of a
  for (std::size_t i = 0; i < num_tiles; ++i) {
    graph.setTileMapping(a[i], i); // since we index once, the resulting slices are 2D
  }

  // Apply same tile mapping to b and c
  graph.setTileMapping(b, graph.getTileMapping(a));
  graph.setTileMapping(c, graph.getTileMapping(a));

  // Create compute set object
  auto compute_set = graph.addComputeSet();

  // Assign vertices to this compute set
  for (std::size_t i = 0; i < num_tiles; ++i) {
    // There will be num_worker_contexts (6) vertices per tile (not a requirement)
    for (std::size_t j = 0; j < num_worker_contexts; ++j) {
      // Declare and assign vertex to a tile
      auto v = graph.addVertex(compute_set, vertex);
      graph.connect(v["a"], a[i][j]); // since we index twice, the resulting slices are 1D
      graph.connect(v["b"], b[i][j]); 
      graph.connect(v["c"], c[i][j]);
      graph.setInitialValue(v["q"], q);
      graph.setTileMapping(v, i); // map vertex to tile i
    }
  }

  // Define data streams
  auto device_to_host_a = graph.addDeviceToHostFIFO("a_stream", poplar::FLOAT, a.numElements());
  auto host_to_device_b = graph.addHostToDeviceFIFO("b_stream", poplar::FLOAT, b.numElements());
  auto host_to_device_c = graph.addHostToDeviceFIFO("c_stream", poplar::FLOAT, c.numElements());

  // Poplar program definitions 
  auto host_to_device = poplar::program::Sequence({
    poplar::program::Copy(host_to_device_b, b),
    poplar::program::Copy(host_to_device_c, c)
  });

  auto inner_loop = poplar::program::Repeat(
    num_iterations,
    poplar::program::Sequence{
      poplar::program::WriteUndef(a),
      poplar::program::Execute(compute_set)
    }
  );

  auto device_to_host = poplar::program::Copy(a, device_to_host_a);

  // Bundle programs into a vector
  std::vector<poplar::program::Program> programs{
    host_to_device, 
    inner_loop, 
    device_to_host
  };

  // Compilation of graph
  auto exe = poplar::compileGraph(graph, programs);

  // Use engine to control device executions and data streams
  poplar::Engine engine(std::move(exe));
  engine.connectStream("a_stream", &host_a[0], &host_a[host_a.size()]);
  engine.connectStream("b_stream", &host_b[0], &host_b[host_b.size()]);
  engine.connectStream("c_stream", &host_c[0], &host_c[host_c.size()]);
  engine.load(device);

  // Host to Device
  engine.run(0);

  // Execute compute set (repeatedly) and evalute wall time
  auto start = std::chrono::steady_clock::now();
  engine.run(1);
  auto stop = std::chrono::steady_clock::now();

  // Device to Host
  engine.run(2);

  // Report findings
  auto diff = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
  double time_seconds = 1e-9*diff.count(); // from nano seconds
  float clock_frequency = target.getTileClockFrequency(); // 1330 MHz (GC200 MK2)
  float minimal_bandwidth = num_iterations*total_size*2*sizeof(float); // b and c
  auto theoretical_max_bandwidth = 4*sizeof(float)*clock_frequency*num_tiles;
  float total_squared_error = 0.0, error;
  for (std::size_t i = 0; i < host_a.size(); ++i) {
    error = (host_a[i]) - (host_b[i] + q*host_c[i]); 
    total_squared_error += error*error;
  }

  // Print results
  std::cout <<
      "Vertex:              " << vertex <<
    "\nSize per worker      " << size_per_worker <<
    "\nNumber of IPUs       " << num_ipus <<
    "\nNumber of tiles      " << num_tiles <<
    "\nNumber of iterations " << num_iterations << 
    "\nFLOPS                " << 1e-12*(num_iterations*total_size*2)/time_seconds << " TFLOPS"
    "\nMinimal bandwidth    " << 1e-12*minimal_bandwidth << " TB/s" <<
    "\nTotal squared error  " << total_squared_error << "\n";

  return EXIT_SUCCESS;
}