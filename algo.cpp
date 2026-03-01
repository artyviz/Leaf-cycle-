#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

static constexpr double EPS = 1e-9;

struct Vec3 {
    double x, y, z;
    Vec3(double x = 0, double y = 0, double z = 0) : x(x), y(y), z(z) {}
    Vec3 operator+(const Vec3& o) const { return {x+o.x, y+o.y, z+o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x-o.x, y-o.y, z-o.z}; }
    Vec3 operator*(double s)      const { return {x*s,   y*s,   z*s};   }
    double dot(const Vec3& o)     const { return x*o.x + y*o.y + z*o.z; }
    double norm()                 const { return std::sqrt(dot(*this));  }
    Vec3 normalised()             const { double n=norm()+EPS; return *this*(1.0/n); }
    Vec3 cross(const Vec3& o)     const {
        return {y*o.z - z*o.y, z*o.x - x*o.z, x*o.y - y*o.x};
    }
};

struct Color { double r, g, b; };

struct Leaf {
    int         id;
    Vec3        centroid;
    Vec3        normal;
    double      radius;
    Color       color;
    std::string lifecycle_stage;
    int         depth_layer;
};

struct DBSCAN {
    double eps;
    int    min_pts;

    std::vector<int> run(const std::vector<Vec3>& pts) const {
        int n = (int)pts.size();
        std::vector<int> labels(n, -1);
        int cluster = 0;
        for (int i = 0; i < n; ++i) {
            if (labels[i] != -1) continue;
            auto nb = neighbours(pts, i);
            if ((int)nb.size() < min_pts) { labels[i] = 0; continue; }
            labels[i] = ++cluster;
            for (int j = 0; j < (int)nb.size(); ++j) {
                int q = nb[j];
                if (labels[q] == 0) labels[q] = cluster;
                if (labels[q] != -1) continue;
                labels[q] = cluster;
                auto nb2 = neighbours(pts, q);
                if ((int)nb2.size() >= min_pts)
                    nb.insert(nb.end(), nb2.begin(), nb2.end());
            }
        }
        return labels;
    }

private:
    std::vector<int> neighbours(const std::vector<Vec3>& pts, int i) const {
        std::vector<int> res;
        for (int j = 0; j < (int)pts.size(); ++j)
            if (j != i && (pts[i]-pts[j]).norm() <= eps)
                res.push_back(j);
        return res;
    }
};

Vec3 estimate_normal(const std::vector<Vec3>& pts) {
    if (pts.size() < 3) return {0, 0, 1};
    Vec3 mu{};
    for (auto& p : pts) { mu.x+=p.x; mu.y+=p.y; mu.z+=p.z; }
    mu = mu * (1.0 / pts.size());

    double cxx=0,cxy=0,cxz=0,cyy=0,cyz=0,czz=0;
    for (auto& p : pts) {
        Vec3 d = p - mu;
        cxx+=d.x*d.x; cxy+=d.x*d.y; cxz+=d.x*d.z;
        cyy+=d.y*d.y; cyz+=d.y*d.z; czz+=d.z*d.z;
    }
    Vec3 v{0.1, 0.1, 1.0}; v = v.normalised();
    for (int iter = 0; iter < 50; ++iter) {
        Vec3 Av{ cxx*v.x + cxy*v.y + cxz*v.z,
                 cxy*v.x + cyy*v.y + cyz*v.z,
                 cxz*v.x + cyz*v.y + czz*v.z };
        v = Av.normalised();
    }
    Vec3 t1{1,0,0};
    if (std::abs(v.dot(t1)) > 0.9) t1 = {0,1,0};
    Vec3 t2 = v.cross(t1).normalised();
    return v.cross(t2).normalised();
}

std::string score_lifecycle(const Color& c) {
    double sum = c.r + c.g + c.b + EPS;
    double gs  = c.g / sum;
    double rs  = c.r / sum;
    double val = std::max({c.r, c.g, c.b});
    if (val < 0.20)   return "abscission";
    if (gs > 0.50)    return "expansion_maturity";
    if (gs >= 0.30)   return "senescence";
    if (rs > 0.40)    return "senescence";
    return "bud_emergence";
}

struct DepthOrder {
    bool operator()(const Leaf& a, const Leaf& b) const {
        return a.centroid.z > b.centroid.z;
    }
};

bool aabb_overlap_xy(const Leaf& a, const Leaf& b) {
    return std::abs(a.centroid.x - b.centroid.x) < (a.radius + b.radius) &&
           std::abs(a.centroid.y - b.centroid.y) < (a.radius + b.radius);
}

bool is_occluded_by(const Leaf& back, const Leaf& front) {
    return front.centroid.z > back.centroid.z && aabb_overlap_xy(front, back);
}

struct TreeModel {
    Vec3   canopy_centroid;
    double trunk_radius;
    int    num_layers;
    std::vector<double> layer_radii;
    std::vector<std::vector<int>> layers;

    void build(const std::vector<Leaf>& leaves, int n_layers) {
        if (leaves.empty()) return;
        num_layers = n_layers;

        Vec3 sum{};
        for (auto& l : leaves) sum = sum + l.centroid;
        canopy_centroid = sum * (1.0 / leaves.size());

        double zmin = leaves[0].centroid.z, zmax = leaves[0].centroid.z;
        for (auto& l : leaves) {
            zmin = std::min(zmin, l.centroid.z);
            zmax = std::max(zmax, l.centroid.z);
        }
        double zstep = (zmax - zmin + EPS) / n_layers;

        layers.resize(n_layers);
        layer_radii.resize(n_layers, 0.0);

        for (auto& l : leaves) {
            int k = std::min((int)((l.centroid.z - zmin) / zstep), n_layers-1);
            layers[k].push_back(l.id);
            Vec3 d{ l.centroid.x - canopy_centroid.x,
                    l.centroid.y - canopy_centroid.y, 0 };
            layer_radii[k] = std::max(layer_radii[k], d.norm());
        }

        trunk_radius = *std::max_element(layer_radii.begin(), layer_radii.end()) * 0.05;
    }

    void print() const {
        std::cout << "\n[TreeModel]\n";
        std::printf("  Canopy centroid : (%.3f, %.3f, %.3f)\n",
                    canopy_centroid.x, canopy_centroid.y, canopy_centroid.z);
        std::printf("  Trunk radius    : %.3f\n", trunk_radius);
        for (int k = 0; k < num_layers; ++k)
            std::printf("  Layer %d  – canopy radius %.3f  – %zu leaf/leaves\n",
                        k, layer_radii[k], layers[k].size());
    }
};

int main() {
    std::cout << "==================================================\n";
    std::cout << "  Tree Leaf 3D Identification Algorithm\n";
    std::cout << "==================================================\n\n";

    struct RawPt { double x, y, z, r, g, b; };
    std::vector<RawPt> cloud = {
        {-0.3, -0.2,  0.00, 0.15, 0.70, 0.20},
        {-0.1,  0.1,  0.02, 0.12, 0.68, 0.18},
        { 0.2, -0.1,  0.01, 0.14, 0.72, 0.19},
        { 0.1,  0.3, -0.01, 0.10, 0.65, 0.22},
        { 0.0,  0.0,  0.00, 0.13, 0.71, 0.21},
        {-0.2,  0.0, -1.00, 0.40, 0.52, 0.12},
        { 0.0,  0.2, -0.98, 0.38, 0.50, 0.10},
        { 0.3,  0.1, -1.02, 0.42, 0.51, 0.11},
        { 0.1, -0.2, -1.01, 0.39, 0.49, 0.13},
        { 0.1,  0.1, -1.00, 0.41, 0.53, 0.12},
        { 0.5,  0.5, -3.00, 0.55, 0.30, 0.10},
        { 0.7,  0.6, -2.98, 0.52, 0.28, 0.09},
        { 0.6,  0.4, -3.02, 0.54, 0.31, 0.11},
    };

    std::vector<Vec3>  pts;
    std::vector<Color> colors;
    for (auto& p : cloud) {
        pts.push_back({p.x, p.y, p.z});
        colors.push_back({p.r, p.g, p.b});
    }

    std::cout << "[Step 1] Clustering point cloud with DBSCAN (epsilon=0.5, minPts=3)\n";
    DBSCAN dbscan{0.50, 3};
    auto labels = dbscan.run(pts);

    int max_label = *std::max_element(labels.begin(), labels.end());
    std::printf("  Found %d cluster(s) (noise label = 0)\n\n", max_label);

    std::vector<Leaf> leaves;
    for (int cid = 1; cid <= max_label; ++cid) {
        std::vector<Vec3>  cpts;
        std::vector<Color> ccols;
        for (int i = 0; i < (int)pts.size(); ++i)
            if (labels[i] == cid) { cpts.push_back(pts[i]); ccols.push_back(colors[i]); }

        Vec3 mu{};
        for (auto& p : cpts) { mu.x+=p.x; mu.y+=p.y; mu.z+=p.z; }
        mu = mu * (1.0 / cpts.size());

        Color avg{};
        for (auto& c : ccols) { avg.r+=c.r; avg.g+=c.g; avg.b+=c.b; }
        avg.r/=ccols.size(); avg.g/=ccols.size(); avg.b/=ccols.size();

        double rad = 0;
        for (auto& p : cpts) rad = std::max(rad, (p-mu).norm());

        Vec3 n = estimate_normal(cpts);

        Leaf leaf;
        leaf.id              = cid;
        leaf.centroid        = mu;
        leaf.normal          = n;
        leaf.radius          = rad;
        leaf.color           = avg;
        leaf.lifecycle_stage = score_lifecycle(avg);
        leaf.depth_layer     = 0;
        leaves.push_back(leaf);
    }

    std::cout << "[Step 2] Z-Depth ordering (z=0 is nearest to camera)\n";
    std::sort(leaves.begin(), leaves.end(), DepthOrder{});
    std::printf("  %-6s  %-22s  %-8s  %-22s  %-8s\n",
                "LeafID", "Centroid (x,y,z)", "Depth z", "Normal (a,b,c)", "Stage");
    std::printf("  %s\n", std::string(80,'-').c_str());
    for (auto& l : leaves)
        std::printf("  L%-5d  (%+.3f,%+.3f,%+.3f)   %+.3f    (%+.3f,%+.3f,%+.3f)  %s\n",
                    l.id,
                    l.centroid.x, l.centroid.y, l.centroid.z,
                    l.centroid.z,
                    l.normal.x, l.normal.y, l.normal.z,
                    l.lifecycle_stage.c_str());

    std::cout << "\n[Step 3] Pairwise delta-z analysis\n";
    for (int i = 0; i < (int)leaves.size(); ++i) {
        for (int j = i+1; j < (int)leaves.size(); ++j) {
            double dz = leaves[i].centroid.z - leaves[j].centroid.z;
            bool   occ = is_occluded_by(leaves[j], leaves[i]);
            std::string occ_str = occ
                ? "[L" + std::to_string(leaves[i].id) +
                  " OCCLUDES L" + std::to_string(leaves[j].id) + "]"
                : "";
            std::printf("  L%d -> L%d :  dz = %+.3f   %s\n",
                        leaves[i].id, leaves[j].id, dz, occ_str.c_str());
        }
    }

    std::cout << "\n[Step 4] Building tree canopy model\n";
    TreeModel tree;
    tree.build(leaves, 4);
    tree.print();

    std::cout << "\n==================================================\n";
    std::cout << "  Leaf Inventory\n";
    std::cout << "==================================================\n";
    for (auto& l : leaves) {
        std::printf("  Leaf %d\n", l.id);
        std::printf("    Position      : (%.3f, %.3f, %.3f)\n",
                    l.centroid.x, l.centroid.y, l.centroid.z);
        std::printf("    Depth z       : %.3f\n", l.centroid.z);
        std::printf("    Normal        : (%.3f, %.3f, %.3f)\n",
                    l.normal.x, l.normal.y, l.normal.z);
        std::printf("    Bounding rad  : %.3f\n", l.radius);
        std::printf("    Color (R,G,B) : (%.2f, %.2f, %.2f)\n",
                    l.color.r, l.color.g, l.color.b);
        std::printf("    Lifecycle     : %s\n\n", l.lifecycle_stage.c_str());
    }

    return 0;
}
