#ifndef ALPAKAEVENT_H
#define ALPAKAEVENT_H

namespace cms {
  namespace alpakatools {
    template <typename T_Acc>
    ALPAKA_FN_INLINE auto createEvent(T_Acc const& acc) {
      return alpaka::Event<cms::alpakatools::Queue>(acc);
    }
  }  // namespace alpakatools
}  // namespace cms

#endif  // ALPAKAEVENT_H
