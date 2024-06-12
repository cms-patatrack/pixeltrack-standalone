#ifndef DataFormats_TrackCount_h
#define DataFormats_TrackCount_h

class TrackCount {
public:
  explicit TrackCount(unsigned int n) : tracks_(n) {}

  unsigned int nTracks() const { return tracks_; }

private:
  unsigned int tracks_;
};

#endif
