#pragma once
import Utils.Sampling.SampleGeneratorInterface;

inline float Next<T : ISampleGenerator>(inout T sg)
{
    return sg.next() / 4294967296.0;
}
inline float2 Next2<T : ISampleGenerator>(inout T sg)
{
    return float2(Next(sg), Next(sg));
}
inline float3 Next3<T : ISampleGenerator>(inout T sg)
{
    return float3(Next(sg), Next(sg), Next(sg));
}

inline float radicalInverse_VdC(uint bits)
{
    bits = (bits << 16) | (bits >> 16);
    bits = ((bits & 0x55555555u) << 1) | ((bits & 0xAAAAAAAAu) >> 1);
    bits = ((bits & 0x33333333u) << 2) | ((bits & 0xCCCCCCCCu) >> 2);
    bits = ((bits & 0x0F0F0F0Fu) << 4) | ((bits & 0xF0F0F0F0u) >> 4);
    bits = ((bits & 0x00FF00FFu) << 8) | ((bits & 0xFF00FF00u) >> 8);
    return (float)bits * 2.3283064365386963e-10f; // / 2^32
}

inline float2 Hammersley2D(uint i, uint N)
{
    return float2((i + 0.5f) / (float)N, radicalInverse_VdC(i));
}
