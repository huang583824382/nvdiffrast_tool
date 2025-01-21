import nvdiffrast.torch as dr
import torch
import numpy as np
import trimesh
import os
from PIL import Image

class NvdiffrastRender:
    """nvdiffrastRender based on nvdiffrast."""
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.glctx = dr.RasterizeCudaContext() if torch.cuda.is_available() else dr.RasterizeGLContext()
        self.model = None
        self.texture = None

    def load_model(self, path):
        """Add model to the render.
        Args:
            path (str): model path, support trimesh file types: obj, stl, 3mf ...
        """
        mesh = trimesh.load(path)
        self.model = {
            "vertices": np.array(mesh.vertices, dtype=np.float32),
            "triangles": np.array(mesh.faces, dtype=np.int32),
            "colors": (
                np.array(mesh.visual.vertex_colors[:, :3], dtype=np.float32) / 255.0
                if hasattr(mesh.visual, "vertex_colors")
                else None
            ),
            "uvs": (
                np.array(mesh.visual.uv, dtype=np.float32)
                if hasattr(mesh.visual, "uv")
                else None
            ),
        }
    
    def add_model(self, mesh):
        self.model = {
            "vertices": np.array(mesh.vertices, dtype=np.float32),
            "triangles": np.array(mesh.faces, dtype=np.int32),
            "colors": (
                np.array(mesh.visual.vertex_colors[:, :3], dtype=np.float32) / 255.0
                if hasattr(mesh.visual, "vertex_colors")
                else None
            ),
            "uvs": (
                np.array(mesh.visual.uv, dtype=np.float32)
                if hasattr(mesh.visual, "uv")
                else None
            ),
        }

    def add_texture(self, mtl_path):
        """Add texture to the render.
        Args:
            mtl_path (str): mtl file path.
        """
        base_dir = os.path.dirname(mtl_path)
        with open(mtl_path, "r") as file:
            lines = file.readlines()
            for line in lines:
                if line.startswith("map_Kd"):
                    texture_file = line.split()[-1]
                    texture_path = os.path.join(base_dir, texture_file)
                    try:
                        texture = Image.open(texture_path)
                        self.texture = torch.tensor(
                            np.array(texture), dtype=torch.float32, device=self.device
                        )
                        self.texture = self.texture.flip(0).unsqueeze(0) / 255.0
                    except FileNotFoundError:
                        print(f"Warning: Texture file not found: {texture_path}")
                        self.texture = None

    def clear(self):
        """Clear model and texture in render."""
        self.model = None
        self.texture = None

    def frame(self, k, q, t, h, w, near=0.1, far=1000.0):
        """Render a frame.
        Args:
            k: camera intrinsic matrix fx, fy, cx, cy. (4)
            q: camera quaternion. (4)
            t: camera translation. (3)
            h (int): image height.
            w (int): image width.
        Returns:
            np.array: rendered image.

        """
        assert self.model is not None, "No model loaded"

        # create perspective projection matrix
        fx, fy, cx, cy = k
        proj = self.perspective_projection(fx, fy, cx, cy, near, far, h, w).cuda()

        # create view matrix (w2c)
        R = torch.tensor(self.quaternion_to_matrix(q)).cuda().float()
        T = torch.tensor(t).cuda().float()
        view = torch.eye(4).cuda().float()
        view[:3, :3] = R
        view[:3, 3] = T

        # coordinate system conversion
        convert = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]).cuda().float()
        view = convert @ view 

        vertices = torch.tensor(self.model["vertices"]).cuda()
        triangles = self.model["triangles"]
        colors = self.model["colors"]
        uvs = self.model["uvs"]

        # default color is white
        if colors is None:
            colors = torch.ones((len(vertices), 3), dtype=torch.float32)

        # convert to homogeneous coordinates
        vertices_homogeneous = torch.concatenate(
            [vertices, torch.ones((len(vertices), 1)).cuda()], dim=1
        )

        # transform vertices to clip space
        vertices_cam = vertices_homogeneous @ view.T
        vertices_clip = vertices_cam @ proj.T
        vertices_cam = vertices_cam.unsqueeze(0) # (1, N, 4)
        pos = vertices_clip.unsqueeze(0) # (1, N, 4)
        tri = torch.tensor(triangles, dtype=torch.int32, device=self.device) # (M, 3)
        col = colors.cuda().unsqueeze(0) # (1, N, 3)
        uv = (
                torch.tensor(uvs, dtype=torch.float32, device=self.device).unsqueeze(0)
                if uvs is not None
                else None
            ) # (1, N, 2)

        # rasterize
        rast, rast_db = dr.rasterize(self.glctx, pos, tri, resolution=[h, w])
        if uv is not None and self.texture is not None:
            # texture interpolation
            texc, _ = dr.interpolate(uv, rast, tri)
            out = dr.texture(self.texture.contiguous(), texc)
        else:
            # color interpolation
            out, _ = dr.interpolate(col, rast, tri)

        # mask out background (alpha channel > 0)
        out = torch.where(rast[..., 3:] > 0, out, torch.tensor(1.0).cuda())
        # antialiasing
        out = dr.antialias(out, rast, pos, tri)

        # interpolate depth map from z values
        depth, _ = dr.interpolate(vertices_cam.contiguous(), rast, tri)
        # antialiasing
        depth = dr.antialias(depth, rast, pos, tri)

        rgb = out.cpu().numpy()[0]
        rgb = np.clip(rgb, 0, 1) * 255
        rgb = rgb.astype(np.uint8)
        depth = -depth.cpu().numpy()[0, ..., -2]
        return rgb, depth
    
    @staticmethod
    def perspective_projection(fx, fy, cx, cy, near, far, height, width):
        """Create perspective projection matrix."""
        return torch.tensor(
            [
                [2 * fx / width, 0, (width - 2 * cx) / width, 0],
                [0, 2 * fy / height, -(height - 2 * cy) / height, 0],
                [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)],
                [0, 0, -1, 0],
            ],
            dtype=torch.float32,
        )

    @staticmethod
    def quaternion_to_matrix(q):
        """Convert quaternion to rotation matrix."""
        w, x, y, z = q
        return np.array(
            [
                [
                    1 - 2 * y * y - 2 * z * z,
                    2 * x * y - 2 * z * w,
                    2 * x * z + 2 * y * w,
                ],
                [
                    2 * x * y + 2 * z * w,
                    1 - 2 * x * x - 2 * z * z,
                    2 * y * z - 2 * x * w,
                ],
                [
                    2 * x * z - 2 * y * w,
                    2 * y * z + 2 * x * w,
                    1 - 2 * x * x - 2 * y * y,
                ],
            ]
        )

if __name__ == "__main__":
    # Example usage
    import argparse as arg
    parser = arg.ArgumentParser()
    parser.add_argument("--model", type=str, help="Model path")
    parser.add_argument("--texture", type=str, help="Texture path", default=None)
    parser.add_argument("--size", type=int, nargs=2, help="Image size", default=[512,640])
    parser.add_argument("--k", type=float, nargs=4, help="Intrinsic matrix", default=[650.048,647.183, 324.328, 257.323])
    parser.add_argument("--q", type=float, nargs=4, help="Quaternion w, x, y, z", default=[0.988247,0.0126847,0.125407,0.0864852])
    parser.add_argument("--t", type=float, nargs=3, help="Translation x, y, z", default=[36.7651,15.8478,550.117])
    parser.add_argument("--output_color", type=str, help="Output path", default="output.png")
    parser.add_argument("--ouput_depth", type=str, help="Output depth path", default="output_depth.npy")
    args = parser.parse_args()

    render = NvdiffrastRender()
    render.load_model(args.model)
    if args.texture is not None:
        render.add_texture(args.texture)

    q = args.q
    t = args.t
    k = args.k
    print('q', q)
    print('t', t)
    print('k', k)

    rgb, depth = render.frame(
        k, q, t, args.size[0], args.size[1]
    )

    if args.ouput_depth:
        np.save(args.ouput_depth, depth)
    if args.output_color:
        img = Image.fromarray(rgb)
        img.save(args.output_color)
