#include <poplar/Vertex.hpp>

using namespace poplar;

class AlievPanfilov : public Vertex {
public: 
  AlievPanfilov();

  Vector<Input<Vector<float, VectorLayout::SPAN, 4, false>>> e_in; // padded
  Vector<Output<Vector<float, VectorLayout::SPAN, 4, false>>> e_out;
  Vector<InOut<Vector<float, VectorLayout::SPAN, 4, false>>> r;
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
    for (int i = 1; i < height + 1; ++i) {
      for (int j = 1; j < width + 1; ++j) {
        // Computation of new e
        e_out[i-1][j-1] = e_in[i][j] + dt*(
          d_h2*(-4*e_in[i][j] + e_in[i+1][j] + e_in[i-1][j] + e_in[i][j+1] + e_in[i][j-1]) -
          k*e_in[i][j]*(e_in[i][j] - a)*(e_in[i][j] - 1) - e_in[i][j]*r[i-1][j-1]
        );

        // Computation of new r
        r[i-1][j-1] += dt*(-epsilon - my1*r[i-1][j-1]/(my2 + e_in[i][j]))*
          (r[i-1][j-1] + k*e_in[i][j]*(e_in[i][j] - b - 1));
      }
    }
    return true;
  }
};