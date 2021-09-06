#include <poplar/Vertex.hpp>

using namespace poplar;

class AlievPanfilov : public Vertex {
public: 
  AlievPanfilov();

  Vector<Input<Vector<float, VectorLayout::SPAN, 4, false>>> e_in; // padded (thus +2 in both height and width)
  Vector<Output<Vector<float, VectorLayout::SPAN, 4, false>>> e_out;
  Vector<Input<Vector<float, VectorLayout::SPAN, 4, false>>> r_in;
  Vector<Output<Vector<float, VectorLayout::SPAN, 4, false>>> r_out;
  const int height;
  const int width;
  const float delta;
  const float epsilon;
  const float my1;
  const float my2;
  const float h;
  const float dt;
  const float k;
  const float a;
  const float b;
  
  bool compute () {
    const float d_h2 = delta/(h*h);
    const float minus_epsilon = -epsilon;
    const float b_plus_one = b + 1;
    float rhs; 
    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        // Computation of new e
        int x = i + 1; // i for the padded input
        int y = j + 1; // j for the padded input
        rhs = d_h2*(-4*e_in[x][y] + e_in[x+1][y] + e_in[x-1][y] + e_in[x][y+1] + e_in[x][y-1]); // 6 FLOPS
        rhs -= k*e_in[x][y]*(e_in[x][y] - a)*(e_in[x][y] - 1); // 6 FLOPS
        rhs -= e_in[x][y]*r_in[x][y]; // 2 FLOPS
        e_out[i][j] = e_in[x][y] + rhs*dt; // 2 FLOPS
        // 16 FLOPS so far
        // Computation of new r
        rhs = minus_epsilon - my1*r_in[i][j]/(my2 + e_in[i][j]); // 4 FLOPS
        rhs *= r_in[i][j] + k*e_in[i][j]*(e_in[i][j] - b_plus_one); // 5 FLOPS
        r_out[i][j] = r_in[i][j] + rhs*dt; // 2 FLOPS
        // Total: 27 FLOPS
      }
    }
    return true;
  }
};