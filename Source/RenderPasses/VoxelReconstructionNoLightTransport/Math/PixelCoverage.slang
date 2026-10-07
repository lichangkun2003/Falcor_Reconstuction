#pragma once
#include "GaussianEllipsoid.slang"

// Analytic projected area of a sphere clipped by planes, after transforming
// the voxel-local ellipsoid to a unit sphere. Pixel clipping uses planes
// through the camera. Integrate actual silhouette conic arcs and line segments
// with Green's theorem; no polygon approximation or ray quadrature.
// Geometry inputs remain float; intersections and integrals use double.
static const uint kCoverageMaxPlanes = 12; // Cell + near/far + pixel sides.
static const uint kCoverageMaxArcCuts = 28; // Two roots per plane + facing + endpoints.
static const double kCoveragePi = 3.1415926535897932384626433832795L;
static const double kCoverageTau = 6.283185307179586476925286766559L;

double coverageDot(double3 a, double3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
double coverageDot(double4 a, double4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }
double3 coverageCross(double3 a, double3 b)
{
    return double3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

// DXIL double transcendentals are not available on all supported compilers.
// Range-reduced series avoid silently narrowing angles/integrals to float.
double coverageSqrt(double x)
{
    if (x <= 0.0) return 0.0;
    double scaled = x, factor = 1.0;
    for (uint i = 0; i < 16 && scaled < 5.4210108624275221700372640043497e-20L; ++i)
    {
        scaled *= 18446744073709551616.0L;
        factor *= 2.3283064365386962890625e-10L;
    }
    for (uint i = 0; i < 16 && scaled > 18446744073709551616.0L; ++i)
    {
        scaled *= 5.4210108624275221700372640043497e-20L;
        factor *= 4294967296.0L;
    }
    double seed = double(sqrt(float(scaled)));
    // A float sqrt seed has ~24 bits; two Newton steps exceed double precision.
    [unroll]
    for (uint i = 0; i < 2; ++i) seed = 0.5 * (seed + scaled / seed);
    return seed * factor;
}

void coverageSinCos(double angle, out double sine, out double cosine)
{
    int quadrant = int(floor(float(angle / (0.5 * kCoveragePi) + 0.5)));
    double x = angle - double(quadrant) * (0.5 * kCoveragePi);
    double x2 = x * x;
    double s = x, c = 1.0, st = x, ct = 1.0;
    // Range reduction guarantees |x| <= pi/4. Eight terms put the first
    // omitted term below double roundoff. Unroll to make divisors constant.
    [unroll]
    for (uint i = 1; i <= 8; ++i)
    {
        st *= -x2 / double((2u * i) * (2u * i + 1u));
        ct *= -x2 / double((2u * i - 1u) * (2u * i));
        s += st;
        c += ct;
    }
    uint q = uint((quadrant % 4 + 4) % 4);
    sine = q == 0u ? s : q == 1u ? c : q == 2u ? -s : -c;
    cosine = q == 0u ? c : q == 1u ? -s : q == 2u ? -c : s;
}

double coverageAtan(double value)
{
    double x = abs(value);
    bool reciprocal = x > 1.0;
    if (reciprocal) x = 1.0 / x;
    bool quarterTurn = x > 0.4142135623730950488L;
    if (quarterTurn) x = (x - 1.0) / (x + 1.0);
    double x2 = x * x, term = x, sum = x;
    [unroll]
    for (uint i = 1; i <= 22; ++i)
    {
        term *= -x2;
        sum += term / double(2u * i + 1u);
    }
    if (quarterTurn) sum += 0.25 * kCoveragePi;
    if (reciprocal) sum = 0.5 * kCoveragePi - sum;
    return value < 0.0 ? -sum : sum;
}

double coverageAtan2(double y, double x)
{
    if (x == 0.0) return y > 0.0 ? 0.5 * kCoveragePi : y < 0.0 ? -0.5 * kCoveragePi : 0.0;
    double angle = coverageAtan(y / x);
    if (x < 0.0) angle += y >= 0.0 ? kCoveragePi : -kCoveragePi;
    return angle;
}

double coverageLog(double x)
{
    int exponent = int(floor(log2(float(x))));
    double mantissa = x * double(exp2(float(-exponent)));
    double z = (mantissa - 1.0) / (mantissa + 1.0);
    double z2 = z * z, term = z, sum = z;
    [unroll]
    for (uint i = 1; i <= 22; ++i)
    {
        term *= z2;
        sum += term / double(2u * i + 1u);
    }
    return 2.0 * sum + double(exponent) * 0.693147180559945309417232121458L;
}

double coverageLength(double3 v) { return coverageSqrt(coverageDot(v, v)); }
double3 coverageNormalize(double3 v) { return v / coverageLength(v); }
double coverageCross2(double2 a, double2 b) { return a.x * b.y - a.y * b.x; }

struct CoverageShape
{
    double4 planes[kCoverageMaxPlanes]; // n.u <= b, normalized n.
    uint planeCount;
    bool empty;
    double4 rowX, rowY, rowW; // Homogeneous pixel-local screen coordinates.
    double4 facing; // Sphere normal facing: facing.xyz.u + facing.w.
};

void addCoveragePlane(inout CoverageShape shape, double3 normal, double bound)
{
    double norm = coverageLength(normal);
    if (norm < 1e-30)
    {
        shape.empty = shape.empty || bound < 0.0;
        return;
    }
    normal /= norm;
    bound /= norm;
    if (bound <= -1.0) shape.empty = true; // Empty or a point: zero area.
    if (bound >= 1.0) return;
    // Coincident cell/pixel planes must not contribute an arc twice.
    for (uint i = 0; i < shape.planeCount; ++i)
    {
        double3 delta = normal - shape.planes[i].xyz;
        if (coverageDot(delta, delta) < 1e-24)
        {
            shape.planes[i].w = min(shape.planes[i].w, bound);
            return;
        }
    }
    shape.planes[shape.planeCount++] = double4(normal, bound);
}

bool coverageFeasible(CoverageShape shape, double3 point, int skipPlane = -1)
{
    for (uint i = 0; i < shape.planeCount; ++i)
        if (int(i) != skipPlane && coverageDot(shape.planes[i].xyz, point) > shape.planes[i].w + 2e-12) return false;
    return true;
}

double2 coverageProject(CoverageShape shape, double3 u)
{
    double4 p = double4(u, 1.0);
    return double2(coverageDot(shape.rowX, p), coverageDot(shape.rowY, p)) / coverageDot(shape.rowW, p);
}

double2 coverageProjectDerivative(CoverageShape shape, double3 u, double3 direction)
{
    double4 p = double4(u, 1.0);
    double w = coverageDot(shape.rowW, p);
    double dw = coverageDot(shape.rowW.xyz, direction);
    return double2(coverageDot(shape.rowX.xyz, direction) * w - coverageDot(shape.rowX, p) * dw,
                   coverageDot(shape.rowY.xyz, direction) * w - coverageDot(shape.rowY, p) * dw) / (w * w);
}

void initializeCoverageFacing(inout CoverageShape shape)
{
    double3 x = shape.rowX.xyz, y = shape.rowY.xyz, w = shape.rowW.xyz;
    double3 xy = coverageCross(x, y);
    double determinant = coverageDot(xy, w);
    double3 cameraNumerator = -shape.rowX.w * coverageCross(y, w)
        - shape.rowY.w * coverageCross(w, x) - shape.rowW.w * xy;
    if (coverageLength(w) < 1e-30)
        shape.facing = double4(coverageNormalize(cameraNumerator), 0.0); // Orthographic.
    else
    {
        double sign = determinant >= 0.0 ? 1.0 : -1.0;
        double scale = max(coverageLength(cameraNumerator), abs(determinant));
        shape.facing = double4(sign * cameraNumerator, -abs(determinant)) / scale;
    }
}

double coverageFaceFacing(CoverageShape shape, uint plane)
{
    double4 p = shape.planes[plane];
    double result = coverageDot(p.xyz, shape.facing.xyz) + p.w * shape.facing.w;
    // A plane through the camera projects to a line. Consistently assign it
    // to the front side so degenerate projected faces close the pixel boundary.
    return abs(result) < 2e-12 ? 0.0 : result;
}

bool coveragePlaneLine(double4 a, double4 b, out double3 center, out double3 axis)
{
    double3 crossNormals = coverageCross(a.xyz, b.xyz);
    double determinant = coverageDot(crossNormals, crossNormals);
    center = axis = double3(0.0);
    if (determinant < 1e-28) return false;
    double ab = coverageDot(a.xyz, b.xyz);
    center = ((a.w - ab * b.w) * a.xyz + (b.w - ab * a.w) * b.xyz) / determinant;
    axis = crossNormals / coverageSqrt(determinant);
    return true;
}

bool coverageClipLine(CoverageShape shape, double3 center, double3 axis, out double lo, out double hi)
{
    double radiusSquared = 1.0 - coverageDot(center, center);
    lo = hi = 0.0;
    if (radiusSquared <= 0.0) return false;
    hi = coverageSqrt(radiusSquared);
    lo = -hi;
    for (uint i = 0; i < shape.planeCount; ++i)
    {
        double4 p = shape.planes[i];
        double rate = coverageDot(p.xyz, axis), gap = p.w - coverageDot(p.xyz, center);
        if (abs(rate) < 1e-14)
        {
            if (gap < -2e-12) return false;
        }
        else if (rate > 0.0) hi = min(hi, gap / rate);
        else lo = max(lo, gap / rate);
    }
    return hi > lo;
}

bool coveragePixelCornerHit(CoverageShape shape, double2 corner)
{
    double4 a = shape.rowX - corner.x * shape.rowW;
    double4 b = shape.rowY - corner.y * shape.rowW;
    double na = coverageLength(a.xyz), nb = coverageLength(b.xyz);
    if (na < 1e-30 || nb < 1e-30) return false;
    a = double4(a.xyz / na, -a.w / na);
    b = double4(b.xyz / nb, -b.w / nb);
    double3 center, axis;
    double lo, hi;
    return coveragePlaneLine(a, b, center, axis) && coverageClipLine(shape, center, axis, lo, hi);
}

struct CoverageCircle
{
    double3 center, cosine, sine;
};

CoverageCircle coverageCircle(double3 normal, double bound)
{
    double3 seed = abs(normal.x) < 0.75 ? double3(1, 0, 0) : double3(0, 1, 0);
    double3 tangent = coverageNormalize(coverageCross(normal, seed));
    double radius = coverageSqrt(max(0.0, 1.0 - bound * bound));
    CoverageCircle circle;
    circle.center = bound * normal;
    circle.cosine = radius * tangent;
    circle.sine = radius * coverageCross(normal, tangent);
    return circle;
}

double3 coverageCirclePoint(CoverageCircle circle, double angle)
{
    double sine, cosine;
    coverageSinCos(angle, sine, cosine);
    return circle.center + cosine * circle.cosine + sine * circle.sine;
}

void addCoverageArcRoots(CoverageCircle circle, double4 plane, inout double cuts[kCoverageMaxArcCuts], inout uint count)
{
    double a = coverageDot(plane.xyz, circle.cosine), b = coverageDot(plane.xyz, circle.sine);
    double c = plane.w - coverageDot(plane.xyz, circle.center);
    double radius = coverageSqrt(a * a + b * b);
    if (radius < 1e-25 || abs(c) >= radius) return;
    double phase = coverageAtan2(b, a);
    double offset = coverageAtan2(coverageSqrt(max(0.0, radius * radius - c * c)), c);
    for (uint i = 0; i < 2; ++i)
    {
        double angle = phase + (i == 0 ? offset : -offset);
        if (angle < 0.0) angle += kCoverageTau;
        if (angle >= kCoverageTau) angle -= kCoverageTau;
        uint insertion = count++;
        while (insertion > 0 && cuts[insertion - 1] > angle)
        {
            cuts[insertion] = cuts[insertion - 1];
            --insertion;
        }
        cuts[insertion] = angle;
    }
}

// J0=integral(1/W^2), J1=integral(1/W), W=a+rho*cos(t).
// Closed-form evaluation for elliptic, hyperbolic and parabolic conics.
void coverageRationalIntegrals(double a, double rho, double ta, double tb, out double j0, out double j1)
{
    double sa, ca, sb, cb;
    coverageSinCos(ta, sa, ca);
    coverageSinCos(tb, sb, cb);
    double wa = a + rho * ca, wb = a + rho * cb;
    double delta = a * a - rho * rho;
    double turn = double(int(floor(float((0.5 * (ta + tb) + kCoveragePi) / kCoverageTau))));
    double shiftedA = ta - turn * kCoverageTau, shiftedB = tb - turn * kCoverageTau;
    double sha, cha, shb, chb;
    coverageSinCos(0.5 * shiftedA, sha, cha);
    coverageSinCos(0.5 * shiftedB, shb, chb);
    double ua = sha / cha, ub = shb / chb;
    double capitalA = a + rho, capitalB = a - rho;
    double epsilon = capitalB / capitalA;
    bool noPole = shiftedA > -kCoveragePi + 1e-12 && shiftedB < kCoveragePi - 1e-12;
    if (noPole && capitalA > 0.0 && abs(epsilon) * max(ua * ua, ub * ub) < 0.125)
    {
        // Stable analytic series near a parabolic conic, including delta=0.
        // Terms fall below double roundoff after 24 iterations. No geometry
        // sampling or numerical boundary quadrature is introduced.
        j0 = j1 = 0.0;
        double termA = ua, termB = ub;
        double ratioA = -epsilon * ua * ua, ratioB = -epsilon * ub * ub;
        for (uint k = 0; k < 24; ++k)
        {
            double odd = double(2u * k + 1u);
            double integral = (termB - termA) / odd;
            double nextIntegral = (termB * ub * ub - termA * ua * ua) / (odd + 2.0);
            j1 += integral;
            j0 += double(k + 1u) * (integral + nextIntegral);
            termA *= ratioA;
            termB *= ratioB;
        }
        j1 *= 2.0 / capitalA;
        j0 *= 2.0 / (capitalA * capitalA);
        return;
    }
    if (delta > 0.0)
    {
        double root = coverageSqrt(delta);
        double angleA = coverageAtan2(root * sa, a * ca + rho);
        double angleB = coverageAtan2(root * sb, a * cb + rho);
        double sweep = angleB - angleA;
        if (sweep < 0.0) sweep += kCoverageTau;
        if (tb - ta > kCoverageTau - 1e-12) sweep = kCoverageTau;
        j1 = sweep / root;
    }
    else
    {
        double root = coverageSqrt(-delta);
        double scale = coverageSqrt(-capitalB / capitalA);
        double va = scale * ua, vb = scale * ub;
        j1 = coverageLog(((1.0 + vb) * (1.0 - va)) / ((1.0 - vb) * (1.0 + va))) / root;
    }
    j0 = (a * j1 - rho * (sb / wb - sa / wa)) / delta;
}

double coverageArcGreenIntegral(CoverageShape shape, CoverageCircle circle, double lo, double hi, bool projectedLine)
{
    if (projectedLine)
        return 0.5 * coverageCross2(coverageProject(shape, coverageCirclePoint(circle, lo)),
                                  coverageProject(shape, coverageCirclePoint(circle, hi)));
    double3 x = double3(coverageDot(shape.rowX, double4(circle.center, 1.0)),
        coverageDot(shape.rowX.xyz, circle.cosine), coverageDot(shape.rowX.xyz, circle.sine));
    double3 y = double3(coverageDot(shape.rowY, double4(circle.center, 1.0)),
        coverageDot(shape.rowY.xyz, circle.cosine), coverageDot(shape.rowY.xyz, circle.sine));
    double3 w = double3(coverageDot(shape.rowW, double4(circle.center, 1.0)),
        coverageDot(shape.rowW.xyz, circle.cosine), coverageDot(shape.rowW.xyz, circle.sine));
    double k0 = x.y * y.z - x.z * y.y;
    double kc = x.x * y.z - x.z * y.x;
    double ks = x.y * y.x - x.x * y.y;
    double rho = coverageSqrt(w.y * w.y + w.z * w.z);
    if (rho < 1e-15 * abs(w.x))
    {
        double sl, cl, sh, ch;
        coverageSinCos(lo, sl, cl);
        coverageSinCos(hi, sh, ch);
        return 0.5 * (k0 * (hi - lo) + kc * (sh - sl) - ks * (ch - cl)) / (w.x * w.x);
    }
    double phase = coverageAtan2(w.z, w.y);
    double cp = w.y / rho, sp = w.z / rho;
    double cosineCoefficient = kc * cp + ks * sp;
    double sineCoefficient = -kc * sp + ks * cp;
    double j0, j1;
    double ta = lo - phase, tb = hi - phase;
    coverageRationalIntegrals(w.x, rho, ta, tb, j0, j1);
    double delta = w.x * w.x - rho * rho;
    if (abs(delta) > 1e-6 * (w.x * w.x + rho * rho))
    {
        // Centered-conic form avoids subtracting J1-a*J0 and dividing by a
        // tiny rho when the camera is far away (almost orthographic).
        double2 conicCenter = (w.x * double2(x.x, y.x) - w.y * double2(x.y, y.y)
            - w.z * double2(x.z, y.z)) / delta;
        double determinant = w.x * k0 - w.y * kc - w.z * ks;
        double2 endpointDelta = coverageProject(shape, coverageCirclePoint(circle, hi))
            - coverageProject(shape, coverageCirclePoint(circle, lo));
        return 0.5 * (determinant * j1 / delta + coverageCross2(conicCenter, endpointDelta));
    }
    double sa, ca, sb, cb;
    coverageSinCos(ta, sa, ca);
    coverageSinCos(tb, sb, cb);
    double jCos = (j1 - w.x * j0) / rho;
    double jSin = (1.0 / (w.x + rho * cb) - 1.0 / (w.x + rho * ca)) / rho;
    return 0.5 * (k0 * j0 + cosineCoefficient * jCos + sineCoefficient * jSin);
}

double integrateCoverageCircle(CoverageShape shape, CoverageCircle circle, int owner, bool frontFace)
{
    double cuts[kCoverageMaxArcCuts];
    cuts[0] = 0.0;
    cuts[1] = kCoverageTau;
    uint count = 2;
    for (uint i = 0; i < shape.planeCount; ++i)
        if (int(i) != owner) addCoverageArcRoots(circle, shape.planes[i], cuts, count);
    double4 visibilityPlane = double4(shape.facing.xyz, -shape.facing.w);
    if (owner >= 0)
    {
        if (!frontFace) visibilityPlane = -visibilityPlane;
        addCoverageArcRoots(circle, visibilityPlane, cuts, count);
    }
    bool projectedLine = owner >= 0 && coverageFaceFacing(shape, uint(owner)) == 0.0;
    double area = 0.0;
    for (uint i = 0; i + 1 < count; ++i)
    {
        double lo = cuts[i], hi = cuts[i + 1];
        if (hi - lo < 1e-13) continue;
        double mid = 0.5 * (lo + hi);
        double sine, cosine;
        coverageSinCos(mid, sine, cosine);
        double3 point = circle.center + cosine * circle.cosine + sine * circle.sine;
        if (!coverageFeasible(shape, point, owner)) continue;
        if (owner >= 0 && coverageDot(visibilityPlane.xyz, point) > visibilityPlane.w + 2e-12) continue;
        if (coverageDot(shape.rowW, double4(point, 1.0)) <= 0.0) continue;
        double3 tangent = -sine * circle.cosine + cosine * circle.sine;
        double3 inward = -point;
        if (owner >= 0) inward -= shape.planes[owner].xyz;
        double orientation = coverageCross2(coverageProjectDerivative(shape, point, tangent),
                                             coverageProjectDerivative(shape, point, inward));
        if (abs(orientation) < 1e-28) continue;
        double integral = coverageArcGreenIntegral(shape, circle, lo, hi, projectedLine);
        area += orientation > 0.0 ? integral : -integral;
    }
    return area;
}

double integrateCoverageBoundary(CoverageShape shape)
{
    if (shape.empty) return 0.0;
    // Projection of this clipped convex body is convex. Four contained pixel
    // corners imply coverage exactly one; avoid all boundary integration.
    if (coveragePixelCornerHit(shape, double2(-0.5, -0.5)) && coveragePixelCornerHit(shape, double2(0.5, -0.5))
        && coveragePixelCornerHit(shape, double2(0.5, 0.5)) && coveragePixelCornerHit(shape, double2(-0.5, 0.5)))
        return 1.0;
    double area = 0.0;
    double normalLength = coverageLength(shape.facing.xyz);
    double contourBound = normalLength > 0.0 ? -shape.facing.w / normalLength : 2.0;
    if (contourBound < 1.0)
    {
        CoverageCircle contour = coverageCircle(shape.facing.xyz / normalLength, contourBound);
        area += integrateCoverageCircle(shape, contour, -1, false);
    }
    for (uint i = 0; i < shape.planeCount; ++i)
    {
        double4 plane = shape.planes[i];
        bool front = coverageFaceFacing(shape, i) >= 0.0;
        CoverageCircle circle = coverageCircle(plane.xyz, plane.w);
        double circleArea = integrateCoverageCircle(shape, circle, int(i), front);
        area += circleArea;
        for (uint j = i + 1; j < shape.planeCount; ++j)
        {
            bool otherFront = coverageFaceFacing(shape, j) >= 0.0;
            if (front == otherFront) continue;
            double3 center, axis;
            double lo, hi;
            bool lineExists = coveragePlaneLine(plane, shape.planes[j], center, axis);
            bool segmentExists = lineExists && coverageClipLine(shape, center, axis, lo, hi);
            if (!segmentExists) continue;
            double3 a = center + lo * axis, b = center + hi * axis;
            double3 midpoint = 0.5 * (a + b);
            if (coverageDot(shape.rowW, double4(midpoint, 1.0)) <= 0.0) continue;
            double3 inward = -plane.xyz - shape.planes[j].xyz;
            double orientation = coverageCross2(coverageProjectDerivative(shape, midpoint, axis),
                                                 coverageProjectDerivative(shape, midpoint, inward));
            double integral = 0.5 * coverageCross2(coverageProject(shape, a), coverageProject(shape, b));
            double edgeArea = orientation > 0.0 ? integral : -integral;
            area += edgeArea;
        }
    }
    return min(1.0, max(0.0, area));
}

float clippedEllipsoidPixelCoverage(
    GaussianEllipsoid ellipsoid, int3 cell, float3 gridMinimum, float3 cellSize,
    float4x4 vp, uint2 resolution, uint2 pixel)
{
    double3 scale = double3(gaussianEllipsoidScale(ellipsoid));
    float3x3 rotationFloat = gaussianRotationMatrix(ellipsoid.rotation);
    double3x3 rotation = double3x3(rotationFloat);
    CoverageShape shape;
    shape.empty = false;
    shape.planeCount = 0;
    for (uint i = 0; i < 3; ++i)
    {
        double3 row = rotation[i] * scale / double(cellSize[i]);
        addCoveragePlane(shape, row, 1.0 - double(ellipsoid.center[i]));
        addCoveragePlane(shape, -row, double(ellipsoid.center[i]));
    }
    double3 centerWorld = double3(gridMinimum) + (double3(cell) + double3(ellipsoid.center)) * double3(cellSize);
    double4 centerClip;
    double4x3 clipBasis;
    for (uint i = 0; i < 4; ++i)
    {
        centerClip[i] = coverageDot(double4(vp[i]), double4(centerWorld, 1.0));
        for (uint j = 0; j < 3; ++j)
            clipBasis[i][j] = coverageDot(double3(vp[i].xyz), double3(rotation[0][j], rotation[1][j], rotation[2][j]) * scale[j]);
    }
    addCoveragePlane(shape, -clipBasis[2], centerClip.z);
    addCoveragePlane(shape, clipBasis[2] - clipBasis[3], centerClip.w - centerClip.z);
    shape.rowW = double4(clipBasis[3], centerClip.w);
    shape.rowX = 0.5 * double(resolution.x) * double4(clipBasis[0], centerClip.x)
        + (0.5 * double(resolution.x) - double(pixel.x) - 0.5) * shape.rowW;
    shape.rowY = -0.5 * double(resolution.y) * double4(clipBasis[1], centerClip.y)
        + (0.5 * double(resolution.y) - double(pixel.y) - 0.5) * shape.rowW;
    // Preserve 3D ellipsoid/voxel overlap while clipping its projection to pixel.
    double4 left = -shape.rowX - 0.5 * shape.rowW;
    double4 right = shape.rowX - 0.5 * shape.rowW;
    double4 top = -shape.rowY - 0.5 * shape.rowW;
    double4 bottom = shape.rowY - 0.5 * shape.rowW;
    addCoveragePlane(shape, left.xyz, -left.w);
    addCoveragePlane(shape, right.xyz, -right.w);
    addCoveragePlane(shape, top.xyz, -top.w);
    addCoveragePlane(shape, bottom.xyz, -bottom.w);
    if (shape.empty) return 0.0f;
    initializeCoverageFacing(shape);
    return float(integrateCoverageBoundary(shape));
}
