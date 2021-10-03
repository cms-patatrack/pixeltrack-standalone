#ifndef ALPAKAQUEUE_H
#define ALPAKAQUEUE_H

namespace cms {
  namespace alpakatools {
    template <typename T_Acc>
    ALPAKA_FN_INLINE auto createQueueNonBlocking(T_Acc const& acc) {
      cms::alpakatools::Queue x(acc);
      return x;
    }
  }  // namespace alpakatools
}  // namespace cms

#endif  // ALPAKAQUEUE_H
