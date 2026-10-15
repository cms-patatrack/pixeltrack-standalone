#ifndef ESProducer_h
#define ESProducer_h

namespace edm {
  class EventSetup;

  class ESProducer {
  public:
    ESProducer() = default;
    virtual ~ESProducer() = default;

    virtual void produce(EventSetup& eventSetup) = 0;
  };
}  // namespace edm

#endif
