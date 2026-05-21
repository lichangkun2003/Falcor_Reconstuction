#pragma once
// #include "VoxelizationShared.slang"
#include "../HostDeviceShared.slang"
#include "../Math/Ellipsoid.slang"
#include "../Math/SphericalHarmonics.slang"
#include "../Math/Triangle.slang"
#include "ABSDF.slang"
//#include "VoxelData.slang"


BEGIN_NAMESPACE_FALCOR

#if !defined(HOST_CODE)
#include "../Math/Random.slang"
import Utils.Math.MathHelpers;

const static float Pi = 3.14159265359;
const static float InvPi = 0.318309886183790671538;
const static float MinRough = 0.01;

inline float pow5(float x)
{
    x = saturate(x);
    float x2 = x * x;
    return x2 * x2 * x;
}
inline float3 calcSchlick(float3 f0, float3 f90, float cosTheta)
{
    return lerp(f0, f90, pow5(1 - cosTheta));
}
inline float calcSchlick(float f0, float f90, float cosTheta)
{
    return lerp(f0, f90, pow5(1 - cosTheta));
}

inline float ggx(float NdotH, float alpha)
{
    float alpha2 = alpha * alpha;
    float temp = NdotH * NdotH * (alpha2 - 1) + 1;
    temp = max(1e-6, temp);
    return alpha2 / Pi / temp / temp;
}
inline float3 fresnel(float3 v, float3 h, float3 spec)
{
    return calcSchlick(spec, float3(1, 1, 1), dot(v, h));
}

float Lambda(float NdotX, float alpha)
{
    float temp = alpha * alpha + (1 - alpha * alpha) * NdotX * NdotX;
    temp = safeSqrt(temp) / max(1e-6, NdotX) - 1;
    return temp / 2;
}

inline float3 sampleGGX_VNDF(float3 v, float alpha, float2 u)
{
    // 1) stretch view
    float3 Vh = normalize(float3(alpha * v.x, alpha * v.y, v.z));

    // 2) build orthonormal basis around Vh
    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    float3 T1 = (lensq > 0.0f) ? float3(-Vh.y, Vh.x, 0.0f) * rsqrt(lensq) : float3(1, 0, 0);
    float3 T2 = cross(Vh, T1);

    // 3) uniform sample on disk (no trig)
    float2 p = sample_disk_concentric(u);
    float t1 = p.x;
    float t2 = p.y;

    // 4) Heitz “visible normals” warp
    float s = 0.5f * (1.0f + Vh.z);
    t2 = lerp(safeSqrt(max(0.0f, 1.0f - t1 * t1)), t2, s);

    // 5) reproject to hemisphere
    float3 Nh = t1 * T1 + t2 * T2 + safeSqrt(max(0.0f, 1.0f - t1 * t1 - t2 * t2)) * Vh;

    // 6) unstretch
    return normalize(float3(alpha * Nh.x, alpha * Nh.y, max(0.0f, Nh.z)));
}

#endif



struct FrostbiteCosBRDF
{
    float3 diffuse; // Lambert反射率
    float rough;

#if !defined(HOST_CODE)
    [mutating]
    void init(float3 diffuse, float rough)
    {
        this.diffuse = diffuse;
        this.rough = rough;
    }

    float3 eval(float3 l, float3 v, float3 h)
    {
        if (l.z <= 0 || v.z <= 0)
            return 0;
        float VdotH = dot(v, h);
        float energyBias = lerp(0.f, 0.5f, rough);
        float energyFactor = lerp(1.f, 1.f / 1.51f, rough);
        float fd90 = energyBias + 2.f * VdotH * VdotH * rough;
        float fd0 = 1.f;
        float wiScatter = calcSchlick(fd0, fd90, l.z);
        float woScatter = calcSchlick(fd0, fd90, v.z);
        return wiScatter * woScatter * energyFactor * diffuse / Pi * l.z;
    }

    void sample<T : ISampleGenerator>(float3 v, out float3 l, out float3 h, inout T sg)
    {
        float2 d = sample_disk_concentric(Next2(sg));
        float z = safeSqrt(1.f - dot(d, d));
        l = float3(d, z);
        h = normalize(l + v);
    }

    float evalPdf(float3 l, float3 v, float3 h)
    {
        if (l.z <= 0 || v.z <= 0)
            return 0;
        return InvPi * l.z;
    }
#endif
};

struct CookTorrenceCosBRDF
{
    float3 specular; // Fresnel反射率
    float alpha;

#if !defined(HOST_CODE)
    [mutating]
    void init(float3 specular, float rough)
    {
        this.specular = specular;
        alpha = rough * rough;
    }

    float3 eval(float3 l, float3 v, float3 h)
    {
        if (l.z <= 0 || v.z <= 0)
            return 0.f;
        float3 specColor = ggx(h.z, alpha) * fresnel(v, h, specular) / (1 + Lambda(l.z, alpha) + Lambda(v.z, alpha)) / 4 / max(1e-6, v.z);
        return specColor;
    }

    void sample<T : ISampleGenerator>(float3 v, out float3 l, out float3 h, inout T sg)
    {
        float2 u = Next2(sg);
        h = sampleGGX_VNDF(v, alpha, u);
        float VdotH = dot(v, h);
        l = reflect(-v, h);
    }

    float evalPdf(float3 l, float3 v, float3 h)
    {
        if (l.z <= 0 || v.z <= 0)
            return 0;
        float VdotH = dot(v, h);
        if (VdotH <= 0)
            return 0;
        float D = ggx(h.z, alpha);
        float G1 = 1.0 / (1.0 + Lambda(v.z, alpha));
        return D * G1 * h.z / (4.0 * max(1e-6, v.z));
    }
#endif
};

// 从此结构体开始接收和返回世界空间方向，法线分布是中心对称的，只存了上半球面的Lobe
struct LobeBRDF
{
    FrostbiteCosBRDF frostBite;
    CookTorrenceCosBRDF cookTorrence;

    float3 t, b, n;
    float specularWeight;

#if !defined(HOST_CODE)
    float evalSpecularWeight()
    {
        float3 specular = cookTorrence.specular;
        float specularWeight = (20 * (specular.x + specular.y + specular.z) / 3 + 1) / 21.0;
        return saturate(specularWeight);
    }

    float evalBRDFWeight()
    {
        float glossy = (1 - frostBite.rough) * evalSpecularWeight();
        return lerp(0.2, 0.8, glossy);
    }

    [mutating]
    void init(ABSDFLobe lobe)
    {
        n = lobe.normal;
        buildTBN(t, b, n);
        float rough = max(lobe.rough, MinRough);
        frostBite.init(lobe.diffuse, rough);
        cookTorrence.init(lobe.specular, rough);
        specularWeight = evalSpecularWeight();
    }

    [mutating]
    void update(float3 normal) {
        n = normal;
        buildTBN(t, b, n);
    }

    // 只有l,v在平面两侧时，才会返回0
    float3 eval(float3 l, float3 v, float3 h)
    {
        bool negative = dot(v, n) < 0;
        l = worldToLocal(l, t, b, n);
        v = worldToLocal(v, t, b, n);
        h = worldToLocal(h, t, b, n);
        if (negative)
        {
            l = -l;
            v = -v;
            h = -h;
        }
        return frostBite.eval(l, v, h) + cookTorrence.eval(l, v, h);
    }

    void sample<T : ISampleGenerator>(float3 v, out float3 l, out float3 h, inout T sg)
    {
        bool negative = dot(v, n) < 0;
        v = worldToLocal(v, t, b, n);
        if (negative)
            v = -v;

        float uEnergy = Next(sg);
        if (uEnergy <= specularWeight)
            cookTorrence.sample(v, l, h, sg);
        else
            frostBite.sample(v, l, h, sg);

        l = localToWorld(l, t, b, n);
        h = localToWorld(h, t, b, n);
        if (negative)
        {
            l = -l;
            h = -h;
        }
    }
    // 只有l,v在平面两侧时，才会返回0
    float evalPdf(float3 l, float3 v, float3 h)
    {
        bool negative = dot(v, n) < 0;
        l = worldToLocal(l, t, b, n);
        v = worldToLocal(v, t, b, n);
        h = worldToLocal(h, t, b, n);
        if (negative)
        {
            l = -l;
            v = -v;
            h = -h;
        }
        return (1 - specularWeight) * frostBite.evalPdf(l, v, h) + specularWeight * cookTorrence.evalPdf(l, v, h);
    }
#endif
};

struct SurfaceBRDF
{
    LobeBRDF lobes[LOBE_COUNT];
    float weights[LOBE_COUNT];
    uint lobeCount;

#if !defined(HOST_CODE)
    [mutating]
    void init(ABSDF ABSDF)
    {
        lobeCount = 0;
        for (uint i = 0; i < LOBE_COUNT; i++)
        {
            ABSDFLobe lobe = ABSDF.lobes[i];
            if (lobe.IsValid())
            {
                lobes[lobeCount].init(lobe);
                weights[lobeCount] = lobe.weight;
                lobeCount++;
            }
        }
        if (lobeCount == 0)
        {
            // 防止没有有效lobes
            ABSDFLobe defaultLobe;
            defaultLobe.init();
            defaultLobe.normal = float3(0, 0, 1);
            lobes[0].init(defaultLobe);
            weights[0] = 1;
            lobeCount = 1;
        }
    }

    uint getRandomLobe(float uLobe)
    {
        for (uint i = 0; i < lobeCount; i++)
        {
            uLobe -= weights[i];
            if (uLobe <= 0)
                return i;
        }
        return lobeCount - 1;
    }

    float calcBRDFWeight()
    {
        float sum = 0;
        for (uint i = 0; i < lobeCount; i++)
        {
            sum += weights[i] * lobes[i].evalBRDFWeight();
        }
        return sum;
    }

    // 可视化法线分布时，要渲染的点在球面上的方向充当h即可
    inline float evalNormalDistribution(float3 h)
    {
        float sum = 0;
        for (int i = 0; i < lobeCount; i++)
        {
            float3 n = lobes[i].n;
            float NdotH = dot(n, h);
            if (NdotH < 0)
            {
                n = -n;
                NdotH = -NdotH;
            }
            float alpha = lobes[i].cookTorrence.alpha;
            sum += weights[i] * ggx(NdotH, alpha);
        }
        return sum;
    }

    float3 eval(float3 l, float3 v, float3 h)
    {
        float3 sum = 0;
        for (uint i = 0; i < lobeCount; i++)
        {
            sum += weights[i] * lobes[i].eval(l, v, h);
        }
        return sum;
    }

    void sample<T : ISampleGenerator>(float3 v, out float3 l, out float3 h, inout T sg)
    {
        float uLobe = Next(sg);
        uint index = getRandomLobe(uLobe);
        return lobes[index].sample(v, l, h, sg);
    }

    float evalPdf(float3 l, float3 v, float3 h)
    {
        float pdf = 0;
        for (uint i = 0; i < lobeCount; i++)
        {
            pdf += weights[i] * lobes[i].evalPdf(l, v, h);
        }
        return pdf;
    }

    // 返回与所给方向夹角不超过90°且权重最大的法线
    float3 getMainNormal(float3 direction)
    {
        float3 n = 0;
        float maxWeight = 0;
        for (int i = 0; i < lobeCount; i++)
        {
            if (weights[i] > maxWeight)
            {
                n = lobes[i].n;
                float cosTerm = dot(n, direction);
                if (cosTerm < 0)
                    n = -n;
                maxWeight = weights[i];
            }
        }
        return n;
    }
#endif
};

#if !defined(HOST_CODE)
struct DeltaBTDF
{
    float3 eval(float3 l, float3 v, float3 h) { return 1; }

    void sample<T : ISampleGenerator>(float3 v, out float3 l, out float3 h, inout T sg)
    {
        l = -v;
        h = 0;
    }

    float evalPdf(float3 l, float3 v, float3 h)
    {
        return 1; // 实质上是无穷大
    }
};
#endif

struct PrimitiveBSDF
{
    SurfaceBRDF surface;
#if !defined(HOST_CODE)
    DeltaBTDF delta; // GPU上，没有成员变量的结构体不占空间(人为添加无意义的成员变量也会被优化掉)
#endif
    SphericalFunc primitiveProjAreaFunc;
    SphericalFunc polygonsProjAreaFunc; // 所有多边形的可见投影面积
    SphericalFunc totalProjAreaFunc;    // 所有多边形的总投影面积
    float area;

    float coverage; // 临时变量

#if !defined(HOST_CODE)
    //[mutating]
    //void init(VoxelData data)
    //{
    //    area = data.ABSDF.area;
    //    if (area > 0)
    //    {
    //        primitiveProjAreaFunc = data.primitiveProjAreaFunc;
    //        polygonsProjAreaFunc = data.polygonsProjAreaFunc;
    //        totalProjAreaFunc = data.totalProjAreaFunc;
    //        surface.init(data.ABSDF);
    //    }
    //}

    float calcCoverage(float3 direction)
    {
#if !CHECK_COVERAGE
        return 1;
#endif
        float primitiveProjArea = primitiveProjAreaFunc.calc(direction);
        if (primitiveProjArea <= 0)
            return 1;
        float polygonsProjArea = polygonsProjAreaFunc.calc(direction);
        return saturate(polygonsProjArea / primitiveProjArea);
    }
    // 采样前必须调用此函数
    [mutating]
    void updateCoverage(float3 direction, float threshould)
    {
#if !CHECK_COVERAGE
        coverage = 1;
        return;
#endif
        coverage = calcCoverage(direction);
        if (1 - coverage < threshould)
            coverage = 1;
    }

    // 自遮挡导致入射光减少
    float calcInternalVisibility(float3 direction)
    {
#if !CHECK_VISIBILITY
        return 1.f;
#endif
        float totalProjArea = totalProjAreaFunc.calc(direction);
        if (totalProjArea <= 0)
            return 1;
        float polygonsProjArea = polygonsProjAreaFunc.calc(direction);
        return saturate(polygonsProjArea / totalProjArea);
    }

    void sample<T : ISampleGenerator>(float3 v, out float3 l, out float3 h, inout T sg)
    {
        if (coverage == 1 || Next(sg) <= coverage)
            surface.sample(v, l, h, sg);
        else
            delta.sample(v, l, h, sg);
    }

    float3 eval(float3 l, float3 v, float3 h)
    {
        if (coverage < 1 && all(h == 0))
            return (1 - coverage) * delta.eval(l, v, h);
        return coverage * surface.eval(l, v, h) * calcInternalVisibility(l);
    }

    float evalPdf(float3 l, float3 v, float3 h)
    {
        if (coverage < 1 && all(h == 0))
            return (1 - coverage) * delta.evalPdf(l, v, h);
        return coverage * surface.evalPdf(l, v, h);
    }
#endif
};

struct PrimitiveBSDF_New
{
    SurfaceBRDF surface;
#if !defined(HOST_CODE)
    DeltaBTDF delta; // GPU上，没有成员变量的结构体不占空间(人为添加无意义的成员变量也会被优化掉)
#endif
    //SphericalFunc primitiveProjAreaFunc;
    //SphericalFunc polygonsProjAreaFunc; // 所有多边形的可见投影面积
    //SphericalFunc totalProjAreaFunc;    // 所有多边形的总投影面积

    SphericalFunc coverageFunc;
    SphericalFunc visibilityFunc;
    float area;

    float coverage; // 临时变量

#if !defined(HOST_CODE)
    //[mutating]
    // void init(VoxelData data)
    //{
    //    area = data.ABSDF.area;
    //    if (area > 0)
    //    {
    //        primitiveProjAreaFunc = data.primitiveProjAreaFunc;
    //        polygonsProjAreaFunc = data.polygonsProjAreaFunc;
    //        totalProjAreaFunc = data.totalProjAreaFunc;
    //        surface.init(data.ABSDF);
    //    }
    //}

    float calcCoverage(float3 direction)
    {
#if !CHECK_COVERAGE
        return 1;
#endif
        float couverage = coverageFunc.calc(direction);
        return saturate(couverage);
    }
    // 采样前必须调用此函数
    [mutating]
    void updateCoverage(float3 direction, float threshould)
    {
#if !CHECK_COVERAGE
        coverage = 1;
        return;
#endif
        coverage = calcCoverage(direction);
        if (1 - coverage < threshould)
            coverage = 1;
    }

    // 自遮挡导致入射光减少
    float calcInternalVisibility(float3 direction)
    {
#if !CHECK_VISIBILITY
        return 1.f;
#endif
        float visibility = visibilityFunc.calc(direction);
        return saturate(visibility);
    }

    void sample<T : ISampleGenerator>(float3 v, out float3 l, out float3 h, inout T sg)
    {
        if (coverage == 1 || Next(sg) <= coverage)
            surface.sample(v, l, h, sg);
        else
            delta.sample(v, l, h, sg);
    }

    float3 eval(float3 l, float3 v, float3 h)
    {
        if (coverage < 1 && all(h == 0))
            return (1 - coverage) * delta.eval(l, v, h);
        return coverage * surface.eval(l, v, h) * calcInternalVisibility(l);
    }

    float evalPdf(float3 l, float3 v, float3 h)
    {
        if (coverage < 1 && all(h == 0))
            return (1 - coverage) * delta.evalPdf(l, v, h);
        return coverage * surface.evalPdf(l, v, h);
    }
#endif
};




END_NAMESPACE_FALCOR
