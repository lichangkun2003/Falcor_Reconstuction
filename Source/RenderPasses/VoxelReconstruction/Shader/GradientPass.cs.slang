#include "../GradRecord.slang"
#include "../PathRecord.slang"
//#include "../Voxel/Shading.slang"
#include "../Voxel/VoxelGrid.slang"

import Utils.Debug.PixelDebug;


// Input
Texture2D<float4> gDL_dColorBuffer;
StructuredBuffer<PathRecord> gPathRecordBuffer;


// Output
RWStructuredBuffer<GradRecord> gGradBuffer;
RWTexture2D<float4> dummy;

cbuffer CB {
    uint2 gResolution;
    uint3 gVoxelCount;
    float trasmittanceThreshold;
}

#define INTERLOCKED_ADD_FLOAT(DEST, VALUE)                                                                             \
    {                                                                                                                  \
        float expected_ = DEST;                                                                                        \
        float original_ = 0.0f;                                                                                        \
        [allow_uav_condition]                                                                                          \
        for (uint iter_ = 0; iter_ < 256; iter_++)                                                                     \
        {                                                                                                              \
            float desired_ = expected_ + (VALUE);                                                                      \
            InterlockedCompareExchangeFloatBitwise(                     \
            DEST,                                                   \
            expected_,                                               \
            desired_,                                                \
            original_                                                \
        );                              \
            if (asuint(original_) == asuint(expected_))                                                                \
                break;                                                                                                 \
            expected_ = original_;                                                                                     \
        }                                                                                                              \
    }



float3 calc_dC_dPrimitiveBSDF_New(float3 Li, float pdf)
{
    return Li  / max(pdf, 1e-6f);
}

float calc_dPrimitiveBSDF_New_dSurfaceBRDF(float coverage,float internalVisibility,float3 h )
{
    bool isDeltaBranch = (coverage < 1.0f && all(h == 0));
    if (isDeltaBranch)
        return 0;
    else
        return  coverage * internalVisibility;
}

float3 calc_dSurfaceBRDF_dWeight(
    SurfaceBRDF surface,
    uint i,
    float3 l,
    float3 v,
    float3 h
)
{
    if (i >= surface.lobeCount)
        return float3(0.0f);

    return surface.lobes[i].eval(l, v, h);
}

float calc_dSurfaceBRDF_dLobe(SurfaceBRDF surface,uint i)
{
    if (i >= surface.lobeCount)
        return 0.0f;

    return surface.weights[i];
}

float calc_dLobe_dDiffuseCoeff(LobeBRDF lobe,float3 l,float3 v,float3 h)
{
    bool negative = dot(v, lobe.n) < 0.0f;

    float3 lLocal = worldToLocal(l, lobe.t, lobe.b, lobe.n);
    float3 vLocal = worldToLocal(v, lobe.t, lobe.b, lobe.n);
    float3 hLocal = worldToLocal(h, lobe.t, lobe.b, lobe.n);

    if (negative)
    {
        lLocal = -lLocal;
        vLocal = -vLocal;
        hLocal = -hLocal;
    }

    if (lLocal.z <= 0.0f || vLocal.z <= 0.0f)
        return 0.0f;

    FrostbiteCosBRDF frostBite = lobe.frostBite;

    float rough = frostBite.rough;
    float VdotH = dot(vLocal, hLocal);

    float energyBias = lerp(0.0f, 0.5f, rough);
    float energyFactor = lerp(1.0f, 1.0f / 1.51f, rough);

    float fd90 = energyBias + 2.0f * VdotH * VdotH * rough;
    float fd0 = 1.0f;

    float wiScatter = calcSchlick(fd0, fd90, lLocal.z);
    float woScatter = calcSchlick(fd0, fd90, vLocal.z);

    return wiScatter * woScatter * energyFactor * InvPi * lLocal.z;
}

float3 calc_dLobe_dSpecular(LobeBRDF lobe,float3 l,float3 v,float3 h)
{
    bool negative = dot(v, lobe.n) < 0.0f;

    float3 lLocal = worldToLocal(l, lobe.t, lobe.b, lobe.n);
    float3 vLocal = worldToLocal(v, lobe.t, lobe.b, lobe.n);
    float3 hLocal = worldToLocal(h, lobe.t, lobe.b, lobe.n);

    if (negative)
    {
        lLocal = -lLocal;
        vLocal = -vLocal;
        hLocal = -hLocal;
    }

    if (lLocal.z <= 0.0f || vLocal.z <= 0.0f)
        return float3(0.0f);

    CookTorrenceCosBRDF cook = lobe.cookTorrence;

    float alpha = cook.alpha;

    float VdotH = dot(vLocal, hLocal);
    if (VdotH <= 0.0f)
        return float3(0.0f);

    float D = ggx(hLocal.z, alpha);

    float lambdaL = Lambda(lLocal.z, alpha);
    float lambdaV = Lambda(vLocal.z, alpha);

    float denom = 1.0f + lambdaL + lambdaV;
    denom = max(denom, 1e-6f);

    float common =
        D
        / denom
        / 4.0f
        / max(1e-6f, vLocal.z);

    // fresnel(v, h, specular)
    // = calcSchlick(specular, float3(1,1,1), VdotH)
    //
    // Schlick:
    // F = specular + (1 - specular) * (1 - VdotH)^5
    //   = specular * (1 - s) + s
    //
    // dF/dspecular = 1 - s
    float s = pow5(1.0f - VdotH);

    float dF_dSpecular = 1.0f - s;

    return common * dF_dSpecular;
}


float3 calc_dFrostbite_dRough(FrostbiteCosBRDF frostBite,float3 l,float3 v,float3 h)
{
    if (l.z <= 0.0f || v.z <= 0.0f)
        return float3(0.0f);

    float rough = frostBite.rough;
    float3 diffuse = frostBite.diffuse;

    float VdotH = dot(v, h);
    float VdotH2 = VdotH * VdotH;

    float energyBias = lerp(0.0f, 0.5f, rough);
    float energyFactor = lerp(1.0f, 1.0f / 1.51f, rough);

    float dEnergyFactor_dRough = (1.0f / 1.51f) - 1.0f;

    // fd90 = energyBias + 2 * VdotH^2 * rough
    // energyBias = 0.5 * rough
    float fd90 = energyBias + 2.0f * VdotH2 * rough;
    float dFd90_dRough = 0.5f + 2.0f * VdotH2;

    float fd0 = 1.0f;

    float wiScatter = calcSchlick(fd0, fd90, l.z);
    float woScatter = calcSchlick(fd0, fd90, v.z);

    // calcSchlick(fd0, fd90, cos)
    // = fd0 + (fd90 - fd0) * (1 - cos)^5
    float dWiScatter_dRough = dFd90_dRough * pow5(1.0f - l.z);
    float dWoScatter_dRough = dFd90_dRough * pow5(1.0f - v.z);

    float scalar =
        (
            dWiScatter_dRough * woScatter * energyFactor +
            wiScatter * dWoScatter_dRough * energyFactor +
                    wiScatter * woScatter * dEnergyFactor_dRough
        )
        * InvPi
        * l.z;

    return scalar * diffuse;
}

// dCookTorrence_dAlpha--------------------------------------------------------------------
float dGGX_dAlpha(float NdotH, float alpha)
{
    float a = max(alpha, 1e-6f);
    float a2 = a * a;

    float n = NdotH;
    float n2 = n * n;

    float temp = n2 * (a2 - 1.0f) + 1.0f;
    temp = max(1e-6f, temp);

    float dTemp_dAlpha = 2.0f * a * n2;

    float dD =
        2.0f * a / (Pi * temp * temp)
        - a2 * 2.0f * dTemp_dAlpha / (Pi * temp * temp * temp);

    return dD;
}

float dLambda_dAlpha(float NdotX, float alpha)
{
    float a = max(alpha, 1e-6f);
    float x = max(NdotX, 1e-6f);

    float x2 = x * x;

    float inside = a * a + (1.0f - a * a) * x2;
    inside = max(inside, 1e-6f);

    float sqrtInside = sqrt(inside);

    return a * (1.0f - x2) / (2.0f * x * sqrtInside);
}

float3 calc_dCookTorrence_dAlpha(CookTorrenceCosBRDF cookTorrence,float3 l,float3 v,float3 h)
{
    if (l.z <= 0.0f || v.z <= 0.0f)
        return float3(0.0f);

    float alpha = max(cookTorrence.alpha, 1e-6f);
    float3 specular = cookTorrence.specular;

    float VdotH = dot(v, h);
    if (VdotH <= 0.0f)
        return float3(0.0f);

    float D = ggx(h.z, alpha);
    float dD_dAlpha = dGGX_dAlpha(h.z, alpha);

    float lambdaL = Lambda(l.z, alpha);
    float lambdaV = Lambda(v.z, alpha);

    float dLambdaL_dAlpha = dLambda_dAlpha(l.z, alpha);
    float dLambdaV_dAlpha = dLambda_dAlpha(v.z, alpha);

    float denom = 1.0f + lambdaL + lambdaV;
    denom = max(denom, 1e-6f);

    float dDenom_dAlpha = dLambdaL_dAlpha + dLambdaV_dAlpha;

    float invFactor = 1.0f / (4.0f * max(1e-6f, v.z));

    float3 F = fresnel(v, h, specular);

    // Cook = D * F / denom * invFactor
    //
    // dCook/dAlpha =
    // F * invFactor * d(D / denom)/dAlpha
    //
    // d(D / denom) =
    // (dD * denom - D * dDenom) / denom^2
    float scalar =
        (dD_dAlpha * denom - D * dDenom_dAlpha)
        / (denom * denom)
        * invFactor;

    return F * scalar;
}
//-----------------------------------------------------------------------------------------

float3 calc_dLobe_dRough(LobeBRDF lobe,float3 l,float3 v,float3 h){
    bool negative = dot(v, lobe.n) < 0.0f;

    float3 lLocal = worldToLocal(l, lobe.t, lobe.b, lobe.n);
    float3 vLocal = worldToLocal(v, lobe.t, lobe.b, lobe.n);
    float3 hLocal = worldToLocal(h, lobe.t, lobe.b, lobe.n);

    if (negative)
    {
        lLocal = -lLocal;
        vLocal = -vLocal;
        hLocal = -hLocal;
    }

    if (lLocal.z <= 0.0f || vLocal.z <= 0.0f)
        return float3(0.0f);

    float rough = lobe.frostBite.rough;

    // Frostbite 直接使用 rough
    float3 dFrostbite_dRough = calc_dFrostbite_dRough(
        lobe.frostBite,
        lLocal,
        vLocal,
        hLocal
    );

    // CookTorrence 使用 alpha = rough * rough
    float3 dCook_dAlpha = calc_dCookTorrence_dAlpha(
        lobe.cookTorrence,
        lLocal,
        vLocal,
        hLocal
    );

    float dAlpha_dRough = 2.0f * rough;

    return dFrostbite_dRough + dCook_dAlpha * dAlpha_dRough;
}

// dLobe_dNomrl
float3 evalLobeWithNormal(LobeBRDF lobe,float3 n,float3 l,float3 v,float3 h)
{
    n = normalize(n);

    float3 t;
    float3 b;
    buildTBN(t, b, n);

    lobe.n = n;
    lobe.t = t;
    lobe.b = b;

    return lobe.eval(l, v, h);
}

void calc_dLobe_dNormal_FiniteDifference(
    LobeBRDF lobe,
    float3 l,
    float3 v,
    float3 h,
    out float3 dLobe_dnX,
    out float3 dLobe_dnY,
    out float3 dLobe_dnZ
)
{
    float eps = 1e-3f;

    float3 n0 = normalize(lobe.n);

    float3 base = evalLobeWithNormal(lobe, n0, l, v, h);

    float3 nx = normalize(n0 + float3(eps, 0.0f, 0.0f));
    float3 ny = normalize(n0 + float3(0.0f, eps, 0.0f));
    float3 nz = normalize(n0 + float3(0.0f, 0.0f, eps));

    float3 fx = evalLobeWithNormal(lobe, nx, l, v, h);
    float3 fy = evalLobeWithNormal(lobe, ny, l, v, h);
    float3 fz = evalLobeWithNormal(lobe, nz, l, v, h);

    dLobe_dnX = (fx - base) / eps;
    dLobe_dnY = (fy - base) / eps;
    dLobe_dnZ = (fz - base) / eps;
}

// dL/dCoverage
float calcSaturateDerivativeMask(float rawValue)
{
    if (rawValue <= 0.0f || rawValue >= 1.0f)
        return 0.0f;

    return 1.0f;
}

float calcCoverageDerivativeMask(float rawCoverage, float coverage, float threshold)
{
    if (rawCoverage <= 0.0f || rawCoverage >= 1.0f)
        return 0.0f;

    // 对应 updateCoverage:
    // if (1 - coverage < threshold)
    //     coverage = 1;
    //
    // 这段截断处先按不可导处理
    if (1.0f - coverage < threshold)
        return 0.0f;

    return 1.0f;
}

float3 calc_dPrimitiveBSDF_New_dCoverage(
    PrimitiveBSDF_New BSDF,
    float coverage,
    float internalVisibility,
    float3 l,
    float3 v,
    float3 h
)
{
    bool isDeltaBranch = (coverage < 1.0f && dot(h, h) < 1e-8f);

    if (isDeltaBranch)
    {
        // Primitive = (1 - coverage) * delta
        // dPrimitive/dCoverage = -delta
        return -BSDF.delta.eval(l, v, h);
    }

    // Primitive = coverage * surface.eval(l,v,h) * visibility
    // dPrimitive/dCoverage = surface.eval(l,v,h) * visibility
    float3 surfaceVal = BSDF.surface.eval(l, v, h);
    return surfaceVal * internalVisibility;
}


// dL/dVisibility
float3 calc_dPrimitiveBSDF_New_dVisibility(
    PrimitiveBSDF_New BSDF,
    float coverage,
    float3 l,
    float3 v,
    float3 h
)
{
    bool isDeltaBranch = (coverage < 1.0f && dot(h, h) < 1e-8f);

    if (isDeltaBranch)
        return float3(0.0f);

    // Primitive = coverage * surface.eval(l,v,h) * visibility
    return coverage * BSDF.surface.eval(l, v, h);
}

// theta Gradient
void calc_dL_dTheta(PathRecord rec, float3 dL_dC, out GradRecord gradRecord)
{
    gradRecord.init();

    if (rec.valid == 0)
        return ;

    float3 l = rec.lightDirPdf.xyz;
    float pdf = rec.lightDirPdf.w;
    float3 Li = rec.lightColor.rgb;
    float3 v = rec.viewDir.xyz;
    float3 h = normalizeSafe(l + v);

    if (all(h == float3(0.0f)))
        return;

    if (pdf <= 0.0f)
        return ;


    uint voxelId = CellToIndex(rec.hitCellInt, gGridDataParamBlock.voxelCount);

    PrimitiveBSDF_New BSDF = gGridDataParamBlock.gridDataBuffer[voxelId].bsdf;

    //if (lobeId >= BSDF.surface.lobeCount)
    //    return float3(0.0f);

    // 与 forward 保持一致
    BSDF.updateCoverage(v, trasmittanceThreshold);

    float coverage = BSDF.coverage;

    float internalVisibility = BSDF.calcInternalVisibility(l);


    float3 dC_dPrimitive =
        calc_dC_dPrimitiveBSDF_New(Li, pdf);

    float dPrimitive_dSurface =
        calc_dPrimitiveBSDF_New_dSurfaceBRDF(
            coverage,
            internalVisibility,
            h
        );

    // Coverage Gradient
#if CHECK_COVERAGE
    {
        float rawCoverage = BSDF.coverageFunc.calc(v);

        float coverageMask = calcCoverageDerivativeMask(
        rawCoverage,
        coverage,
        trasmittanceThreshold
    );

        float3 dPrimitive_dCoverage =
        calc_dPrimitiveBSDF_New_dCoverage(
            BSDF,
            coverage,
            internalVisibility,
            l,
            v,
            h
        );

        float3 dC_dCoverage =
        dC_dPrimitive * dPrimitive_dCoverage;

        float gradCoverageScalar =
        dot(dL_dC, dC_dCoverage) * coverageMask;

        for (uint i = 0; i < EVEN_SH_COUNT; i++)
        {
            gradRecord.coverageFunc.coefficients[i] =
            gradCoverageScalar * calcEvenSH(i, v);
        }
    }
#endif

    // Visibility Gradient
#if CHECK_VISIBILITY
    {
        float rawVisibility = BSDF.visibilityFunc.calc(l);

        float visibilityMask =
        calcSaturateDerivativeMask(rawVisibility);

        float3 dPrimitive_dVisibility =
        calc_dPrimitiveBSDF_New_dVisibility(
            BSDF,
            coverage,
            l,
            v,
            h
        );

        float3 dC_dVisibility =
        dC_dPrimitive * dPrimitive_dVisibility;

        float gradVisibilityScalar =
        dot(dL_dC, dC_dVisibility) * visibilityMask;

        for (uint i = 0; i < EVEN_SH_COUNT; i++)
        {
            gradRecord.visibilityFunc.coefficients[i] =
            gradVisibilityScalar * calcEvenSH(i, l);
        }
    }
#endif



    // Lobe Gradient
    for (uint lobeId = 0; lobeId < LOBE_COUNT; lobeId++) {

        if (lobeId >= BSDF.surface.lobeCount)
            break;

        float dSurface_dLobe =
        calc_dSurfaceBRDF_dLobe(BSDF.surface, lobeId);

        // -------------------------
        // lobe wight gradient
        // -------------------------
        float3 dSurface_dWeight = calc_dSurfaceBRDF_dWeight(BSDF.surface, lobeId, l, v, h);

        float3 dC_dWeight =
            dC_dPrimitive
            * dPrimitive_dSurface
            * dSurface_dWeight;

        // -------------------------
        // diffuse gradient
        // -------------------------
        float dLobe_dDiffuse =
        calc_dLobe_dDiffuseCoeff(
            BSDF.surface.lobes[lobeId],
            l,
            v,
            h
        );

        float3 dC_dDiffuse =
        dC_dPrimitive
        * dPrimitive_dSurface
        * dSurface_dLobe
        * dLobe_dDiffuse;


        // -------------------------
        // rough gradient
        // -------------------------
        float3 dLobe_dRough =
            calc_dLobe_dRough(
                BSDF.surface.lobes[lobeId],
                l,
                v,
                h
            );

        float3 dC_dRough = dC_dPrimitive
        * dPrimitive_dSurface
        * dSurface_dLobe
        * dLobe_dRough;

        // -------------------------
        // specular gradient
        // -------------------------
        float3 dLobe_dSpecular =
        calc_dLobe_dSpecular(
            BSDF.surface.lobes[lobeId],
            l,
            v,
            h
        );

        float3 dC_dSpecular =
            dC_dPrimitive
            * dPrimitive_dSurface
            * dSurface_dLobe
            * dLobe_dSpecular;

        // -------------------------
        // normal gradient by finite difference
        // -------------------------
        float3 dLobe_dnX;
        float3 dLobe_dnY;
        float3 dLobe_dnZ;

        calc_dLobe_dNormal_FiniteDifference(
            BSDF.surface.lobes[lobeId],
            l,
            v,
            h,
            dLobe_dnX,
            dLobe_dnY,
            dLobe_dnZ
        );

        float3 common =
            dC_dPrimitive
            * dPrimitive_dSurface
            * dSurface_dLobe;

        float3 dC_dnX = common * dLobe_dnX;
        float3 dC_dnY = common * dLobe_dnY;
        float3 dC_dnZ = common * dLobe_dnZ;

        float3 gradNormal;
        gradNormal.x = dot(dL_dC, dC_dnX);
        gradNormal.y = dot(dL_dC, dC_dnY);
        gradNormal.z = dot(dL_dC, dC_dnZ);

        // normal 是单位向量，梯度投影到切平面
        float3 n = normalize(BSDF.surface.lobes[lobeId].n);
        gradNormal = gradNormal - n * dot(gradNormal, n);

        // Record
        gradRecord.diffuse[lobeId] = dL_dC * dC_dDiffuse;
        gradRecord.specular[lobeId] = dL_dC * dC_dSpecular;
        gradRecord.rough[lobeId] = dot(dL_dC, dC_dRough);
        gradRecord.weight[lobeId] = dot(dL_dC, dC_dWeight);
        gradRecord.normal[lobeId] = gradNormal;
    }
    gradRecord.valid = 1;
}

//bool InterlockedAddFloat(uint indexGradBuffer, float valueToAdd, out float oldValue, out float newValue)
//{
//    float expected = gGradBuffer[indexGradBuffer].test;
//    float original = 0.0f;

//    [allow_uav_condition]
//    for (uint iter = 0; iter < 64; iter++)
//    {
//        float desired = expected + valueToAdd;

//        InterlockedCompareExchangeFloatBitwise(
//            gGradBuffer[indexGradBuffer].test,
//            expected,
//            desired,
//            original
//        );

//        if (asuint(original) == asuint(expected))
//        {
//            oldValue = original;
//            newValue = desired;
//            return true;
//        }

//        expected = original;
//    }

//    oldValue = expected;
//    newValue = expected;
//    return false;
//}

void accumulateGradRecord(uint indexGradBuffer, GradRecord gradRecord)
{
    InterlockedAdd(gGradBuffer[indexGradBuffer].valid, gradRecord.valid);

    for (uint i = 0; i < EVEN_SH_COUNT; i++)
    {
        INTERLOCKED_ADD_FLOAT(
        gGradBuffer[indexGradBuffer].coverageFunc.coefficients[i],
                              gradRecord.coverageFunc.coefficients[i]
        );

        INTERLOCKED_ADD_FLOAT(
        gGradBuffer[indexGradBuffer].visibilityFunc.coefficients[i],
                              gradRecord.visibilityFunc.coefficients[i]
        );
    }

    for (uint lobeId = 0; lobeId < LOBE_COUNT; lobeId++)
    {
        INTERLOCKED_ADD_FLOAT(
            gGradBuffer[indexGradBuffer].diffuse[lobeId].x,
            gradRecord.diffuse[lobeId].x
        );
        INTERLOCKED_ADD_FLOAT(
            gGradBuffer[indexGradBuffer].diffuse[lobeId].y,
            gradRecord.diffuse[lobeId].y
        );
        INTERLOCKED_ADD_FLOAT(
            gGradBuffer[indexGradBuffer].diffuse[lobeId].z,
            gradRecord.diffuse[lobeId].z
        );

        INTERLOCKED_ADD_FLOAT(
            gGradBuffer[indexGradBuffer].specular[lobeId].x,
            gradRecord.specular[lobeId].x
        );
        INTERLOCKED_ADD_FLOAT(
            gGradBuffer[indexGradBuffer].specular[lobeId].y,
            gradRecord.specular[lobeId].y
        );
        INTERLOCKED_ADD_FLOAT(
            gGradBuffer[indexGradBuffer].specular[lobeId].z,
            gradRecord.specular[lobeId].z
        );

        INTERLOCKED_ADD_FLOAT(
            gGradBuffer[indexGradBuffer].rough[lobeId],
            gradRecord.rough[lobeId]
        );

        INTERLOCKED_ADD_FLOAT(
            gGradBuffer[indexGradBuffer].weight[lobeId],
            gradRecord.weight[lobeId]
        );

        INTERLOCKED_ADD_FLOAT(
            gGradBuffer[indexGradBuffer].normal[lobeId].x,
            gradRecord.normal[lobeId].x
        );
        INTERLOCKED_ADD_FLOAT(
            gGradBuffer[indexGradBuffer].normal[lobeId].y,
            gradRecord.normal[lobeId].y
        );
        INTERLOCKED_ADD_FLOAT(
            gGradBuffer[indexGradBuffer].normal[lobeId].z,
            gradRecord.normal[lobeId].z
        );
    }
}




[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID) {
    printSetPixel(dispatchThreadId.xy);
    uint2 pixel = dispatchThreadId.xy;

    if (pixel.x >= gResolution.x || pixel.y >= gResolution.y)
        return;

    uint indexPathRecordBuffer = pixel.y * gResolution.x + pixel.x;

    PathRecord rec = gPathRecordBuffer[indexPathRecordBuffer];

    if (rec.valid == 0)
        return;

    float3 dL_dC = gDL_dColorBuffer.Load(int3(pixel, 0)).rgb;

    GradRecord gradRecord;
    calc_dL_dTheta(rec, dL_dC, gradRecord);

    // print("gradRecord")
    print("gradRecord.diffuse[0]", gradRecord.diffuse[0]);
    print("gradRecord.diffuse[1]", gradRecord.diffuse[1]);
    print("gradRecord.diffuse[2]", gradRecord.diffuse[2]);
    print("gradRecord.diffuse[3]", gradRecord.diffuse[3]);
    print("dL_dC", dL_dC);

    uint indexGradBuffer = CellToIndex(rec.hitCellInt, gVoxelCount);

    print("Valid", gGradBuffer[indexGradBuffer].valid);

    accumulateGradRecord(indexGradBuffer, gradRecord);



    dummy[pixel] = float4(gradRecord.diffuse[0],1.0f);

}


