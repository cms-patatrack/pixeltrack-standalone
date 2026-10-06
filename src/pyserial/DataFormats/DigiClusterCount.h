#ifndef DataFormats_DigiClusterCount_h
#define DataFormats_DigiClusterCount_h

class DigiClusterCount {
public:
  explicit DigiClusterCount(unsigned int nm, unsigned int nd, unsigned int nc)
      : modules_(nm), digis_(nd), clusters_(nc) {}

  unsigned int nModules() const { return modules_; }
  unsigned int nDigis() const { return digis_; }
  unsigned int nClusters() const { return clusters_; }

private:
  unsigned int modules_, digis_, clusters_;
};

#endif
