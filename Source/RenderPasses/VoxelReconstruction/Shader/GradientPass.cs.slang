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

float3 calc_dSurfaceBRDF_dWeight(SurfaceBRDF surface,uint i,float3 l,float3 v,float3 h){
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

void calc_dLobe_dNormal_FiniteDifference(LobeBRDF lobe,float3 l,float3 v,float3 h,out float3 dLobe_dnX,out float3 dLobe_dnY,out float3 dLobe_dnZ)
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
){
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

// geometry
float3 projectToRayTangent(float3 x, float3 omega)
{
    return x - omega * dot(omega, x);
}

float3 calc_dT_dParam(float3 gradPhi,float dPhi_dParam,float3 omega,float tEval)
{
    float denomPhi = max(dot(gradPhi, gradPhi), 1e-8f);
    float denomT = max(tEval, 1e-6f);

    // dV/dtheta = -gradPhi / ||gradPhi||^2 * dPhi/dtheta
    float3 dV = -gradPhi * (dPhi_dParam / denomPhi);

    // dT/dtheta = 1 / tEval * (I - omega omega^T) * dV
    float3 dT = projectToRayTangent(dV, omega) / denomT;

    return dT;
}


float3 calc_dL_dT_ReparamDirection(PathRecord rec,float3 dL_dC)
{
    // ------------------------------------------------------------
    // 目标：
    //     dL_dT = - dL_dv
    //
    // 当前版本：
    //     primary direct / view-only analytic gradient
    //
    // 包含：
    //     BSDF.eval(l, v, h) 对 v 的解析导数
    //     h = normalize(l + v) 的解析导数
    //     Frostbite diffuse
    //     CookTorrence specular
    //     multi-lobe surface weights
    //
    // 暂不包含：
    //     coverageFunc.calc(v) 对 v 的导数
    //     light dir l 对 hit point 的导数
    //     shadow nested visibility
    // ------------------------------------------------------------

    float3 lWorld = rec.lightDirPdf.xyz;
    float pdf = rec.lightDirPdf.w;
    float3 Li = rec.lightColor.rgb;
    float3 vWorld = normalizeSafe(rec.viewDir.xyz);

    if (pdf <= 0.0f)
        return float3(0.0f);

    if (all(vWorld == float3(0.0f)))
        return float3(0.0f);

    float3 hWorld = normalizeSafe(lWorld + vWorld);
    if (all(hWorld == float3(0.0f)))
        return float3(0.0f);

    uint voxelId = CellToIndex(rec.hitCellInt, gGridDataParamBlock.voxelCount);
    PrimitiveBSDF_New BSDF = gGridDataParamBlock.gridDataBuffer[voxelId].bsdf;

    // 与 forward 保持一致
    BSDF.updateCoverage(vWorld, trasmittanceThreshold);

    // direct lighting:
    // C = Li / pdf * PrimitiveBSDF_New.eval(l, v, h)
    //
    // PrimitiveBSDF_New.eval:
    // if delta branch:
    //     (1 - coverage) * delta
    // else:
    //     coverage * surface.eval(l,v,h) * internalVisibility(l)
    //
    // 当前 direct light 一般不会进入 delta branch，因为 h != 0。
    if (BSDF.coverage < 1.0f && all(hWorld == float3(0.0f)))
        return float3(0.0f);

    float coverage = BSDF.coverage;
    float internalVisibility = BSDF.calcInternalVisibility(lWorld);

    float3 dL_dPrimitive = dL_dC * (Li / max(pdf, 1e-6f));
    float3 dL_dSurface = dL_dPrimitive * coverage * internalVisibility;

    float3 dL_dvWorldTotal = float3(0.0f);

    for (uint lobeId = 0; lobeId < LOBE_COUNT; lobeId++)
    {
        if (lobeId >= BSDF.surface.lobeCount)
            break;

        LobeBRDF lobe = BSDF.surface.lobes[lobeId];

        float3 dL_dLobeEval = dL_dSurface * BSDF.surface.weights[lobeId];

        bool negative = dot(vWorld, lobe.n) < 0.0f;
        float signFlip = negative ? -1.0f : 1.0f;

        float3 l = worldToLocal(lWorld, lobe.t, lobe.b, lobe.n);
        float3 v = worldToLocal(vWorld, lobe.t, lobe.b, lobe.n);

        if (negative)
        {
            l = -l;
            v = -v;
        }

        if (l.z <= 0.0f || v.z <= 0.0f)
            continue;

        float3 hInput = l + v;
        float hInputLen = max(length(hInput), 1e-6f);
        float3 h = hInput / hInputLen;

        if (all(h == float3(0.0f)))
            continue;

        float3 dL_dvLocal = float3(0.0f);

        // ============================================================
        // FrostbiteCosBRDF.eval(l, v, h)
        // ============================================================
        {
            FrostbiteCosBRDF frost = lobe.frostBite;

            float rough = frost.rough;
            float3 diffuse = frost.diffuse;

            float VdotH = dot(v, h);

            float energyBias = lerp(0.0f, 0.5f, rough);
            float energyFactor = lerp(1.0f, 1.0f / 1.51f, rough);

            float fd90 = energyBias + 2.0f * VdotH * VdotH * rough;
            float fd0 = 1.0f;

            float wiScatter = calcSchlick(fd0, fd90, l.z);
            float woScatter = calcSchlick(fd0, fd90, v.z);

            float K = energyFactor * InvPi * l.z;

            // eval = wi * wo * K * diffuse
            float dL_dScalar = dot(dL_dLobeEval, diffuse);

            float dL_dWi = dL_dScalar * woScatter * K;
            float dL_dWo = dL_dScalar * wiScatter * K;

            float dL_dFd90 = 0.0f;
            float dL_dVz = 0.0f;
            float dL_dVdotH = 0.0f;

            // wi = fd0 + (fd90 - fd0) * pow5(1 - l.z)
            float sl = pow5(1.0f - l.z);
            dL_dFd90 += dL_dWi * sl;

            // wo = fd0 + (fd90 - fd0) * pow5(1 - v.z)
            float sv = pow5(1.0f - v.z);
            dL_dFd90 += dL_dWo * sv;

            // d wo / d v.z
            {
                float x = 1.0f - v.z;
                float dsv_dvz = 0.0f;

                if (x > 0.0f && x < 1.0f)
                {
                    float x2 = x * x;
                    dsv_dvz = -5.0f * x2 * x2;
                }

                dL_dVz += dL_dWo * (fd90 - fd0) * dsv_dvz;
            }

            // fd90 = 0.5 * rough + 2 * VdotH^2 * rough
            dL_dVdotH += dL_dFd90 * 4.0f * rough * VdotH;

            // VdotH = dot(v, h)
            float3 dL_dv_fromVdotH = dL_dVdotH * h;
            float3 dL_dh = dL_dVdotH * v;

            dL_dvLocal += dL_dv_fromVdotH;
            dL_dvLocal.z += dL_dVz;

            // h = normalize(l + v)
            float3 dL_dhInput =
                (dL_dh - h * dot(h, dL_dh)) / hInputLen;

            dL_dvLocal += dL_dhInput;
        }

        // ============================================================
        // CookTorrenceCosBRDF.eval(l, v, h)
        // ============================================================
        {
            CookTorrenceCosBRDF cook = lobe.cookTorrence;

            float alpha = max(cook.alpha, 1e-6f);
            float3 specular = cook.specular;

            float VdotH = dot(v, h);
            if (VdotH > 0.0f)
            {
                float NdotH = h.z;
                float NdotV = v.z;
                float NdotL = l.z;

                float D = ggx(NdotH, alpha);
                float lambdaL = Lambda(NdotL, alpha);
                float lambdaV = Lambda(NdotV, alpha);

                float denom = max(1.0f + lambdaL + lambdaV, 1e-6f);
                float invV = 1.0f / (4.0f * max(NdotV, 1e-6f));

                float3 F = fresnel(v, h, specular);

                // eval = D * F / denom / 4 / v.z
                float scalar = D / denom * invV;

                float3 dL_dF = dL_dLobeEval * scalar;
                float dL_dScalar = dot(dL_dLobeEval, F);

                float dL_dD = dL_dScalar / denom * invV;
                float dL_dDenom =
                    dL_dScalar * (-D) / (denom * denom) * invV;

                float dL_dNdotV =
                    dL_dScalar * D / denom
                    * (-1.0f)
                    / max(4.0f * NdotV * NdotV, 1e-6f);

                // Lambda(v.z, alpha)
                {
                    float a = max(alpha, 1e-6f);
                    float a2 = a * a;
                    float x = max(NdotV, 1e-6f);
                    float x2 = x * x;

                    float inside = a2 + (1.0f - a2) * x2;
                    inside = max(inside, 1e-6f);
                    float s = sqrt(inside);

                    // d Lambda / d x
                    float dLambda_dNdotV =
                        -a2 / max(2.0f * s * x2, 1e-6f);

                    dL_dNdotV += dL_dDenom * dLambda_dNdotV;
                }

                // D = ggx(h.z, alpha)
                float dL_dNdotH = 0.0f;
                {
                    float a = max(alpha, 1e-6f);
                    float a2 = a * a;
                    float n = max(NdotH, 1e-6f);
                    float n2 = n * n;

                    float temp = n2 * (a2 - 1.0f) + 1.0f;
                    temp = max(temp, 1e-6f);

                    // dD/dNdotH
                    float dD_dNdotH =
                        -4.0f * a2 * n * (a2 - 1.0f)
                        / (Pi * temp * temp * temp);

                    dL_dNdotH = dL_dD * dD_dNdotH;
                }

                // F = specular + (1 - specular) * pow5(1 - VdotH)
                float3 dF_dVdotH = float3(0.0f);
                {
                    float x = 1.0f - VdotH;
                    float ds_dVdotH = 0.0f;

                    if (x > 0.0f && x < 1.0f)
                    {
                        float x2 = x * x;
                        ds_dVdotH = -5.0f * x2 * x2;
                    }

                    dF_dVdotH =
                        (float3(1.0f, 1.0f, 1.0f) - specular)
                        * ds_dVdotH;
                }

                float dL_dVdotH = dot(dL_dF, dF_dVdotH);

                // VdotH = dot(v, h)
                float3 dL_dv_fromVdotH = dL_dVdotH * h;
                float3 dL_dh = dL_dVdotH * v;

                // h.z
                dL_dh.z += dL_dNdotH;

                // v.z
                dL_dv_fromVdotH.z += dL_dNdotV;

                dL_dvLocal += dL_dv_fromVdotH;

                // h = normalize(l + v)
                float3 dL_dhInput =
                    (dL_dh - h * dot(h, dL_dh)) / hInputLen;

                dL_dvLocal += dL_dhInput;
            }
        }

        // local adjoint -> world adjoint
        // vLocal = signFlip * worldToLocal(vWorld)
        float3 dL_dvWorld =
            signFlip *
            (
                dL_dvLocal.x * lobe.t +
                dL_dvLocal.y * lobe.b +
                dL_dvLocal.z * lobe.n
            );

        dL_dvWorldTotal += dL_dvWorld;
    }

    #if CHECK_COVERAGE
    // 当前版本没有把 coverageFunc.calc(v) 对 v 的导数加进来。
    // 如果之后要补，需要实现 EvenSH 对方向的导数 dSH/dv。
    #endif

    // primary direct:
    //     v = -T
    // 所以：
    //     dL/dT = -dL/dv
    float3 T = -vWorld;
    float3 dL_dT = -dL_dvWorldTotal;

    // T 是单位球方向，只保留切平面分量
    dL_dT = projectToRayTangent(dL_dT, T);

    return dL_dT;
}

float calc_dL_dAreaElement(PathRecord rec,float3 dL_dC){

    float3 directColor = rec.pathContribution.rgb;

    // C_final = pathContribution * areaElement
    // 当前 primal 下 areaElement = 1
    // 所以 dL/dArea = dot(dL/dC, directColor)
    return dot(dL_dC, directColor);
}

float trace3x3(float3x3 M)
{
    return M[0][0] + M[1][1] + M[2][2];
}

float calc_dAreaElement_dParam(
    Ellipsoid ellipsoid,
    float3 xEvalLocal,
    float3 omega,
    float tEval,
    float3 gradPhi,
    float dPhi_dParam,
    float3 d_dPhiParam_dx
)
{
    // ------------------------------------------------------------
    // 解析版 area correction:
    //
    // area_theta = div_omega(dT/dtheta)
    //
    // 当前简化：
    //     固定 tEval，不考虑 dt/domega
    //
    // 令：
    //     g = gradPhi
    //     a = dPhi/dtheta
    //     b = grad_x(dPhi/dtheta)
    //     G = dot(g,g)
    //     H = Hessian(phi) = 2B
    //
    //     A = dV/dtheta = -a * g / G
    //     dT/dtheta = (1/t) * P * A
    //     P = I - omega omega^T
    //
    // 则：
    //     div_S(dT/dtheta)
    //   = trace(P * dA/dx) - 2/t * dot(omega, A)
    //
    // ------------------------------------------------------------

    float t = max(tEval, 1e-6f);

    float3 g = gradPhi;
    float a = dPhi_dParam;
    float3 b = d_dPhiParam_dx;

    float G = max(dot(g, g), 1e-8f);

    float3x3 H = 2.0f * ellipsoid.B;

    float3 Hg = mul(H, g);
    float3 Hu = mul(H, omega);

    float traceH = trace3x3(H);

    // A = -a * g / G
    float3 A = -a * g / G;

    // J_A = dA/dx
    //
    // A = -a g / G
    //
    // dA/dx =
    //   -(g b^T + a H) / G
    //   + 2a / G^2 * g (H g)^T
    //
    // 这里只需要：
    //     trace(P J_A) = trace(J_A) - omega^T J_A omega

    float traceJA =
        -(dot(g, b) + a * traceH) / G
        + (2.0f * a / (G * G)) * dot(g, Hg);

    float3 JA_omega =
        -(g * dot(b, omega) + a * Hu) / G
        + (2.0f * a / (G * G)) * g * dot(Hg, omega);

    float omega_JA_omega = dot(omega, JA_omega);

    float tracePJA = traceJA - omega_JA_omega;

    // div_S((1/t) P A)
    // = trace(P dA/dx) - 2/t * dot(omega, A)
    float divT =
        tracePJA
        - 2.0f * dot(omega, A) / t;

    return divT;
}
float calcGeometryParamGrad(float3 dL_dT, float dL_dArea, Ellipsoid ellipsoid, float3 xEvalLocal, float3 omega, float tEval, float3 gradPhi, float dPhi_dParam,
    float3 d_dPhiParam_dx)
{
    float3 dT_dParam = calc_dT_dParam(
        gradPhi,
        dPhi_dParam,
        omega,
        tEval
    );

    float dArea_dParam = calc_dAreaElement_dParam(
        ellipsoid,
        xEvalLocal,
        omega,
        tEval,
        gradPhi,
        dPhi_dParam,
        d_dPhiParam_dx
    );

    return dot(dL_dT, dT_dParam) + dL_dArea * dArea_dParam;
}

float computeReparamWeight(float score)
{
    // cell-space 下的距离尺度，需要调参
    float sigma = 0.25f;

    // 防止 score 极大导致 exp 数值过小或不稳定
    score = clamp(score, 0.0f, 2.0f);

    return exp(-(score * score) / (2.0f * sigma * sigma));
}

// theta Gradient
void calc_dL_dTheta(PathRecord rec, float3 dL_dC, out GradRecord materialGradRecord, out GradRecord geometryGradRecord)
{
    materialGradRecord.init();
    geometryGradRecord.init();

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
            materialGradRecord.coverageFunc.coefficients[i] =
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
            materialGradRecord.visibilityFunc.coefficients[i] =
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
        materialGradRecord.diffuse[lobeId] = dL_dC * dC_dDiffuse;
        materialGradRecord.specular[lobeId] = dL_dC * dC_dSpecular;
        materialGradRecord.rough[lobeId] = dot(dL_dC, dC_dRough);
        materialGradRecord.weight[lobeId] = dot(dL_dC, dC_dWeight);
        materialGradRecord.normal[lobeId] = gradNormal;
    }
    materialGradRecord.valid = 1;

    // geometry
    if (all(rec.xCellInt >= int3(0)))
    {
        int3 xCellInt = rec.xCellInt;
        uint xVoxelId = CellToIndex(xCellInt, gGridDataParamBlock.voxelCount);

        Ellipsoid ellipsoid =
        gGridDataParamBlock.gridDataBuffer[xVoxelId].ellipsoid;

        // rec.xEval.xyz 当前存的是 grid/cell 坐标
        float3 xEvalGrid = rec.xEval.xyz;
        float3 xEvalLocal = xEvalGrid - float3(xCellInt);

        float reparamScore = rec.xEval.w;
        float reparamWeight = computeReparamWeight(reparamScore);

        float3 rayStart = rec.rayStart.xyz;
        float3 rayVec = xEvalGrid - rayStart;

        float tEval = length(rayVec);
        if (tEval <= 1e-6f)
            return;

        float3 omega = rayVec / tEval;

        float3 q = xEvalLocal - ellipsoid.center;
        float3 Bq = mul(ellipsoid.B, q);

        float3 gradPhi = 2.0f * Bq;
        float gradPhi2 = dot(gradPhi, gradPhi);
        float3x3 H = 2.0f * ellipsoid.B;

        if (gradPhi2 <= 1e-10f)
            return;

        // shading integrand 仍然来自 actual hit voxel

        // 论文式接口：
        float3 dL_dT = calc_dL_dT_ReparamDirection(rec,dL_dC);

        // area element 的上游项
        float dL_dArea = calc_dL_dAreaElement(rec,dL_dC);

        // weighted reparameterization
        dL_dT *= reparamWeight;
        dL_dArea *= reparamWeight;

        // -----------------------------
        // center gradient
        // phi = (x - c)^T B (x - c) - 1
        // dphi/dc = -gradPhi
        // -----------------------------
        float3 dPhi_dCenter = -gradPhi;

        // d/dx (dPhi/dcx) = - d/dx(gradPhi.x)
        float3 d_dPhiCenterX_dx = -float3(H[0][0], H[0][1], H[0][2]);
        float3 d_dPhiCenterY_dx = -float3(H[1][0], H[1][1], H[1][2]);
        float3 d_dPhiCenterZ_dx = -float3(H[2][0], H[2][1], H[2][2]);

        geometryGradRecord.center.x = calcGeometryParamGrad(
            dL_dT,
            dL_dArea,
            ellipsoid,
            xEvalLocal,
            omega,
            tEval,
            gradPhi,
            dPhi_dCenter.x,
            d_dPhiCenterX_dx
        );

        geometryGradRecord.center.y = calcGeometryParamGrad(
            dL_dT,
            dL_dArea,
            ellipsoid,
            xEvalLocal,
            omega,
            tEval,
            gradPhi,
            dPhi_dCenter.y,
            d_dPhiCenterY_dx
        );

        geometryGradRecord.center.z = calcGeometryParamGrad(
            dL_dT,
            dL_dArea,
            ellipsoid,
            xEvalLocal,
            omega,
            tEval,
            gradPhi,
            dPhi_dCenter.z,
            d_dPhiCenterZ_dx
        );

        // -----------------------------
        // B gradient
        //
        // q^T B q =
        // B00 qx^2 + B11 qy^2 + B22 qz^2
        // + 2B01 qxqy + 2B02 qxqz + 2B12 qyqz
        // -----------------------------
        float dPhi_dB00 = q.x * q.x;
        float dPhi_dB11 = q.y * q.y;
        float dPhi_dB22 = q.z * q.z;
        float dPhi_dB01 = 2.0f * q.x * q.y;
        float dPhi_dB02 = 2.0f * q.x * q.z;
        float dPhi_dB12 = 2.0f * q.y * q.z;

        // ∂x(dPhi/dBij)
        float3 d_dPhiB00_dx = float3(2.0f * q.x, 0.0f, 0.0f);
        float3 d_dPhiB11_dx = float3(0.0f, 2.0f * q.y, 0.0f);
        float3 d_dPhiB22_dx = float3(0.0f, 0.0f, 2.0f * q.z);

        float3 d_dPhiB01_dx = float3(2.0f * q.y, 2.0f * q.x, 0.0f);
        float3 d_dPhiB02_dx = float3(2.0f * q.z, 0.0f, 2.0f * q.x);
        float3 d_dPhiB12_dx = float3(0.0f, 2.0f * q.z, 2.0f * q.y);

        geometryGradRecord.B00 = calcGeometryParamGrad(
            dL_dT,
            dL_dArea,
            ellipsoid,
            xEvalLocal,
            omega,
            tEval,
            gradPhi,
            dPhi_dB00,
            d_dPhiB00_dx
        );

        geometryGradRecord.B11 = calcGeometryParamGrad(
            dL_dT,
            dL_dArea,
            ellipsoid,
            xEvalLocal,
            omega,
            tEval,
            gradPhi,
            dPhi_dB11,
            d_dPhiB11_dx
        );

        geometryGradRecord.B22 = calcGeometryParamGrad(
            dL_dT,
            dL_dArea,
            ellipsoid,
            xEvalLocal,
            omega,
            tEval,
            gradPhi,
            dPhi_dB22,
            d_dPhiB22_dx
        );

        geometryGradRecord.B01 = calcGeometryParamGrad(
            dL_dT,
            dL_dArea,
            ellipsoid,
            xEvalLocal,
            omega,
            tEval,
            gradPhi,
            dPhi_dB01,
            d_dPhiB01_dx
        );

        geometryGradRecord.B02 = calcGeometryParamGrad(
            dL_dT,
            dL_dArea,
            ellipsoid,
            xEvalLocal,
            omega,
            tEval,
            gradPhi,
            dPhi_dB02,
            d_dPhiB02_dx
        );

        geometryGradRecord.B12 = calcGeometryParamGrad(
            dL_dT,
            dL_dArea,
            ellipsoid,
            xEvalLocal,
            omega,
            tEval,
            gradPhi,
             dPhi_dB12,
            d_dPhiB12_dx
        );

        geometryGradRecord.geometryValid = 1;
    }

}

void accumulateMaterialGradRecord(uint indexGradBuffer, GradRecord gradRecord)
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

void accumulateGeometryGradRecord(uint indexGradBuffer, GradRecord gradRecord)
{
    InterlockedAdd(gGradBuffer[indexGradBuffer].geometryValid, gradRecord.geometryValid);

    INTERLOCKED_ADD_FLOAT(
        gGradBuffer[indexGradBuffer].center.x,
        gradRecord.center.x
    );
    INTERLOCKED_ADD_FLOAT(
        gGradBuffer[indexGradBuffer].center.y,
        gradRecord.center.y
    );
    INTERLOCKED_ADD_FLOAT(
        gGradBuffer[indexGradBuffer].center.z,
        gradRecord.center.z
    );

    INTERLOCKED_ADD_FLOAT(
        gGradBuffer[indexGradBuffer].B00,
        gradRecord.B00
    );
    INTERLOCKED_ADD_FLOAT(
        gGradBuffer[indexGradBuffer].B11,
        gradRecord.B11
    );
    INTERLOCKED_ADD_FLOAT(
        gGradBuffer[indexGradBuffer].B22,
        gradRecord.B22
    );
    INTERLOCKED_ADD_FLOAT(
        gGradBuffer[indexGradBuffer].B01,
        gradRecord.B01
    );
    INTERLOCKED_ADD_FLOAT(
        gGradBuffer[indexGradBuffer].B02,
        gradRecord.B02
    );
    INTERLOCKED_ADD_FLOAT(
        gGradBuffer[indexGradBuffer].B12,
        gradRecord.B12
    );
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

    GradRecord materialGradRecord;
    GradRecord geometryGradRecord;
    calc_dL_dTheta(rec, dL_dC, materialGradRecord, geometryGradRecord);
    uint indexGradBuffer = CellToIndex(rec.hitCellInt, gVoxelCount);
    accumulateMaterialGradRecord(indexGradBuffer, materialGradRecord);

    // geormetry
    if (all(rec.xCellInt >= int3(0))){
        uint xIndex = CellToIndex(rec.xCellInt, gVoxelCount);
        accumulateGeometryGradRecord(xIndex, geometryGradRecord);
    }
    

    dummy[pixel] = float4(materialGradRecord.diffuse[0],1.0f);

}


