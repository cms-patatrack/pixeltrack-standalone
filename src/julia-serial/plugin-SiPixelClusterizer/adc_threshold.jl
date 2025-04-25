module ADCThreshold
    const  THE_PIXEL_THRESHOLD::Int = 1000;      # default Pixel threshold in electrons
    const  THE_SEED_THRESHOLD::Int = 1000;       # seed thershold in electrons not used in our algo
    const  THE_CLUSTER_THRESHOLD::Float64 = 4000;  # cluster threshold in electron
    const  Conversion_Factor::Int = 65;         # adc to electron conversion factor

    const  THE_STACK_ADC ::Int= 255;               # the maximum adc count for stack layer
    const  THE_FIRST_STACK::Int = 5;               # the index of the fits stack layer
    const  THE_ELECTRON_PAIR_ADC_GAIN::Float64 = 600;  # ADC to electron conversion
end