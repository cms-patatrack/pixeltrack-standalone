#ifndef Framework_ESProducer_h
#define Framework_ESProducer_h

namespace edm {
  class EventSetup;

  class ESProducer {
  public:
    ESProducer() = default;
    virtual ~ESProducer() = default;

    virtual void produce(EventSetup& eventSetup) = 0;
  };
}  // namespace edm

#endif  // Framework_ESProducer_h
