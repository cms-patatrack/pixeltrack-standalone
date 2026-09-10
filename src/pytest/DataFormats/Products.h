#ifndef DataFormats_Products_h
#define DataFormats_Products_h

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace pytest {
  /// The payload of the demo workflow.  The generator's output and both squared
  /// results are this same type, distinguished only by the label of the module
  /// that produced them -- which is what the (type, label) key in
  /// ProductRegistry is for.
  using Floats = std::vector<float>;

  /// The result of comparing two Floats products element by element.
  ///
  /// A user-defined class rather than a tuple of scalars on purpose: it is what
  /// the rootcling/TClass step reflects into nanobind bindings, so the
  /// generator is exercised by this backend and not only by pyserial.
  struct Comparison {
    std::size_t size = 0;          ///< number of elements compared
    std::size_t mismatches = 0;    ///< elements differing by more than the tolerance
    float maxDeviation = 0.0f;     ///< largest absolute difference seen
    float tolerance = 0.0f;        ///< what counted as equal

    bool agree() const { return mismatches == 0; }
  };

  // --------------------------------------------------------------------------
  // Calibration and Samples exist to exercise the binding generator.
  //
  // pyserial's products are the pixel formats, which no test can stand in for;
  // what a test *can* stand in for is their shape.  Between them these two have
  // one of everything the generator emits and every way a Python module reaches
  // the Event -- see D23 in doc/PythonBackends.md for the list -- with
  // arithmetic simple enough that C++ and numpy agree bit for bit.
  //
  // Nothing here is defined out of line: libFramework holds the generated
  // bindings and does not link against libDataFormats, so a constructor in a
  // .cc file would be an undefined symbol there.

  /// A class reached only through a method returning a reference, the shape
  /// SOAFrame has in pyserial (frame.rotation().xx()).
  class Gain {
  public:
    Gain() = default;
    Gain(float slope, float offset) : slope_(slope), offset_(offset) {}

    float slope() const { return slope_; }
    float offset() const { return offset_; }

  private:
    float slope_ = 1.0f;
    float offset_ = 0.0f;
  };

  /// One channel's calibration: a class member, a fixed-size array member and
  /// two scalars, which is what pixelCPEforGPU::DetParams looks like from
  /// Python.
  struct Channel {
    Gain gain;             ///< class member -> a bound reference
    float weights[3];      ///< array member -> a zero-copy numpy view
    std::int32_t index;    ///< scalar member -> def_rw
    bool enabled;
  };

  /// Conditions: an EventSetup product read by type with no token, whose
  /// contents Python reaches both one channel at a time and as columns.
  class Calibration {
  public:
    /// A static constant, reachable from Python without an instance.
    static constexpr std::int32_t kChannels = 8;

    /// The same numbers as columns.  Reading 1856 modules one scalar at a time
    /// is what D20 measured at 35 ns per attribute; a Python module wants the
    /// columns, and a C++ one wants the objects, so a product offers both.
    class View {
    public:
      View() = default;
      View(float const* slope, float const* offset, std::uint8_t const* enabled, std::size_t size)
          : slope_(slope), offset_(offset), enabled_(enabled), size_(size) {}

      std::span<const float> slopeSpan() const { return {slope_, size_}; }
      std::span<const float> offsetSpan() const { return {offset_, size_}; }
      std::span<const std::uint8_t> enabledSpan() const { return {enabled_, size_}; }

    private:
      float const* slope_ = nullptr;
      float const* offset_ = nullptr;
      std::uint8_t const* enabled_ = nullptr;
      std::size_t size_ = 0;
    };

    // The view points into this object's own storage, so a copy would quietly
    // share it; moving is fine, since a vector takes its buffer with it.
    Calibration(Calibration const&) = delete;
    Calibration& operator=(Calibration const&) = delete;
    Calibration(Calibration&&) = default;
    Calibration& operator=(Calibration&&) = default;

    Calibration() : slope_(kChannels), offset_(kChannels), enabled_(kChannels) {
      for (std::int32_t i = 0; i < kChannels; ++i) {
        // Deterministic, and exactly representable in float so that the Python
        // side can be checked against literals.
        const float slope = 1.0f + 0.25f * static_cast<float>(i);
        const float offset = -0.5f * static_cast<float>(i);
        channels_[i].gain = Gain(slope, offset);
        for (int w = 0; w < 3; ++w) {
          channels_[i].weights[w] = 0.5f * static_cast<float>(i) + static_cast<float>(w);
        }
        channels_[i].index = i;
        channels_[i].enabled = (i != 3);  // one dead channel, so `enabled` matters
        slope_[i] = slope;
        offset_[i] = offset;
        enabled_[i] = channels_[i].enabled;
      }
      view_ = View(slope_.data(), offset_.data(), enabled_.data(), kChannels);
    }

    /// A method taking an argument, as PixelCPEFast::detParams(module) is.
    Channel const& channel(std::int32_t i) const {
      if (i < 0 or i >= kChannels) {
        throw std::runtime_error("Calibration: no channel " + std::to_string(i));
      }
      return channels_[i];
    }

    View const& view() const { return view_; }
    std::int32_t channels() const { return kChannels; }

  private:
    Channel channels_[kChannels];
    std::vector<float> slope_;
    std::vector<float> offset_;
    std::vector<std::uint8_t> enabled_;
    View view_;
  };

  /// A product shaped like the SoA formats: columns as spans, a view handed
  /// out by reference, a constructor that wires it to the conditions, and a
  /// method that fills a column once the others have been written.
  class Samples {
  public:
    static constexpr std::uint32_t kMaxSamples = 65536;

    /// The columns.  This is returned by reference on purpose: a method bound
    /// by value hands Python a copy, and everything written through it -- here
    /// `filled` -- is lost.  That bug cost D20 a segmentation fault, and
    /// python/calibrate.py writing through this view is what would catch it.
    class View {
    public:
      View() = default;
      View(float* value, float* scaled, std::int32_t* channel, std::uint32_t size)
          : value_(value), scaled_(scaled), channel_(channel), size_(size) {}

      // Const, and yet the spans are mutable: the view is a handle over storage
      // the product owns, so const on the handle says nothing about the
      // storage -- as with a pointer.  One accessor rather than a const and a
      // non-const overload, because the generator would bind both under the
      // same Python name.
      std::span<float> valueSpan() const { return {value_, size_}; }
      std::span<float> scaledSpan() const { return {scaled_, size_}; }
      std::span<std::int32_t> channelSpan() const { return {channel_, size_}; }

      std::uint32_t size() const { return size_; }

      /// Written by whoever fills the columns.
      std::uint32_t filled = 0;

    private:
      float* value_ = nullptr;
      float* scaled_ = nullptr;
      std::int32_t* channel_ = nullptr;
      std::uint32_t size_ = 0;
    };

    Samples() = default;

    /// `n` samples, calibrated with `calibration`, taken on the channels in
    /// `channel`.  The wiring constructor: a size, another product, and a
    /// column, which is what allocate(token, ...) has to be able to call.
    Samples(std::uint32_t n, Calibration const& calibration, std::span<const std::int32_t> channel)
        : value_(n), scaled_(n), channel_(channel.begin(), channel.end()), calibration_(&calibration) {
      if (channel.size() != n) {
        throw std::runtime_error("Samples: " + std::to_string(n) + " samples but " +
                                 std::to_string(channel.size()) + " channels");
      }
      view_ = View(value_.data(), scaled_.data(), channel_.data(), n);
    }

    // The view holds pointers into this object's own storage, so a copy would
    // quietly share it.  Moving is fine: a vector takes its buffer with it.
    Samples(Samples const&) = delete;
    Samples& operator=(Samples const&) = delete;
    Samples(Samples&&) = default;
    Samples& operator=(Samples&&) = default;

    View& view() { return view_; }
    View const& view() const { return view_; }

    std::uint32_t size() const { return view_.size(); }

    /// Fills `scaled` from `value` and the conditions, after Python has
    /// written `value`: a method with a side effect on the product, the shape
    /// TrackingRecHit2DHeterogeneous::buildIndex() has.
    ///
    /// The arithmetic is a subtraction and a multiplication, never a
    /// multiply-add, so that -march=native cannot contract it into an FMA that
    /// numpy has no way to reproduce.
    void applyCalibration() {
      for (std::uint32_t i = 0; i < view_.size(); ++i) {
        Channel const& channel = calibration_->channel(channel_[i]);
        scaled_[i] = channel.enabled ? (value_[i] - channel.gain.offset()) * channel.gain.slope() : 0.0f;
      }
    }

  private:
    std::vector<float> value_;
    std::vector<float> scaled_;
    std::vector<std::int32_t> channel_;
    Calibration const* calibration_ = nullptr;
    View view_;
  };
}  // namespace pytest

// The header rootcling parses to build the dictionaries for this backend's
// products.  Which of the types declared here are products is said once, in
// REFLECT_PRODUCTS in the Makefile: that list drives the dictionary, the class
// bindings, the get/emplace dispatch and the name -> type_index table alike.
//
// It names std::vector<float> rather than the pytest::Floats alias, because it
// is also the name a Python module asks for and the one ROOT reflected.

#endif
