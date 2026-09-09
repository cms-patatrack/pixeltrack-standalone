#ifndef DataFormats_VertexCount_h
#define DataFormats_VertexCount_h

class VertexCount {
public:
  explicit VertexCount(unsigned int n) : vertices_(n) {}

  unsigned int nVertices() const { return vertices_; }

private:
  unsigned int vertices_;
};

#endif
