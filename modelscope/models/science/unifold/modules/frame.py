# The Uni-fold implementation is also open-sourced by the authors under Apache-2.0 license,
# and is publicly available at https://github.com/dptech-corp/Uni-Fold.

from __future__ import annotations  # noqa
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch


def zero_translation(
    batch_dims: Tuple[int],
    dtype: Optional[torch.dtype] = torch.float,
    device: Optional[torch.device] = torch.device('cpu'),
    requires_grad: bool = False,
) -> torch.Tensor:
    trans = torch.zeros((*batch_dims, 3),
                        dtype=dtype,
                        device=device,
                        requires_grad=requires_grad)
    return trans


# pylint: disable=bad-whitespace
_QUAT_TO_ROT = np.zeros((4, 4, 3, 3), dtype=np.float32)

_QUAT_TO_ROT[0, 0] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # rr
_QUAT_TO_ROT[1, 1] = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]  # ii
_QUAT_TO_ROT[2, 2] = [[-1, 0, 0], [0, 1, 0], [0, 0, -1]]  # jj
_QUAT_TO_ROT[3, 3] = [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]  # kk

_QUAT_TO_ROT[1, 2] = [[0, 2, 0], [2, 0, 0], [0, 0, 0]]  # ij
_QUAT_TO_ROT[1, 3] = [[0, 0, 2], [0, 0, 0], [2, 0, 0]]  # ik
_QUAT_TO_ROT[2, 3] = [[0, 0, 0], [0, 0, 2], [0, 2, 0]]  # jk

_QUAT_TO_ROT[0, 1] = [[0, 0, 0], [0, 0, -2], [0, 2, 0]]  # ir
_QUAT_TO_ROT[0, 2] = [[0, 0, 2], [0, 0, 0], [-2, 0, 0]]  # jr
_QUAT_TO_ROT[0, 3] = [[0, -2, 0], [2, 0, 0], [0, 0, 0]]  # kr

_QUAT_TO_ROT = _QUAT_TO_ROT.reshape(4, 4, 9)
_QUAT_TO_ROT_tensor = torch.from_numpy(_QUAT_TO_ROT)

_QUAT_MULTIPLY = np.zeros((4, 4, 4))
_QUAT_MULTIPLY[:, :, 0] = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0],
                           [0, 0, 0, -1]]

_QUAT_MULTIPLY[:, :, 1] = [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1],
                           [0, 0, -1, 0]]

_QUAT_MULTIPLY[:, :, 2] = [[0, 0, 1, 0], [0, 0, 0, -1], [1, 0, 0, 0],
                           [0, 1, 0, 0]]

_QUAT_MULTIPLY[:, :, 3] = [[0, 0, 0, 1], [0, 0, 1, 0], [0, -1, 0, 0],
                           [1, 0, 0, 0]]

_QUAT_MULTIPLY_BY_VEC = _QUAT_MULTIPLY[:, 1:, :]
_QUAT_MULTIPLY_BY_VEC_tensor = torch.from_numpy(_QUAT_MULTIPLY_BY_VEC)


class Rotation:

    def __init__(
        self,
        mat: torch.Tensor,
    ):
        if mat.shape[-2:] != (3, 3):
            raise ValueError(f'incorrect rotation shape: {mat.shape}')
        self._mat = mat

    @staticmethod
    def identity(
        shape,
        dtype: Optional[torch.dtype] = torch.float,
        device: Optional[torch.device] = torch.device('cpu'),
        requires_grad: bool = False,
    ) -> Rotation:
        mat = torch.eye(
            3, dtype=dtype, device=device, requires_grad=requires_grad)
        mat = mat.view(*((1, ) * len(shape)), 3, 3)
        mat = mat.expand(*shape, -1, -1)
        return Rotation(mat)

    @staticmethod
    def mat_mul_mat(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a.float() @ b.float()).type(a.dtype)

    @staticmethod
    def mat_mul_vec(r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return (r.float() @ t.float().unsqueeze(-1)).squeeze(-1).type(t.dtype)

    def __getitem__(self, index: Any) -> Rotation:
        if not isinstance(index, tuple):
            index = (index, )
        return Rotation(mat=self._mat[index + (slice(None), slice(None))])

    def __mul__(self, right: Any) -> Rotation:
        if isinstance(right, (int, float)):
            return Rotation(mat=self._mat * right)
        elif isinstance(right, torch.Tensor):
            return Rotation(mat=self._mat * right[..., None, None])
        else:
            raise TypeError(
                f'multiplicand must be a tensor or a number, got {type(right)}.'
            )

    def __rmul__(self, left: Any) -> Rotation:
        return self.__mul__(left)

    def __matmul__(self, other: Rotation) -> Rotation:
        new_mat = Rotation.mat_mul_mat(self.rot_mat, other.rot_mat)
        return Rotation(mat=new_mat)

    @property
    def _inv_mat(self):
        return self._mat.transpose(-1, -2)

    @property
    def rot_mat(self) -> torch.Tensor:
        return self._mat

    def invert(self) -> Rotation:
        return Rotation(mat=self._inv_mat)

    def apply(self, pts: torch.Tensor) -> torch.Tensor:
        return Rotation.mat_mul_vec(self._mat, pts)

    def invert_apply(self, pts: torch.Tensor) -> torch.Tensor:
        return Rotation.mat_mul_vec(self._inv_mat, pts)

    # inherit tensor behaviors
    @property
    def shape(self) -> torch.Size:
        s = self._mat.shape[:-2]
        return s

    @property
    def dtype(self) -> torch.dtype:
        return self._mat.dtype

    @property
    def device(self) -> torch.device:
        return self._mat.device

    @property
    def requires_grad(self) -> bool:
        return self._mat.requires_grad

    def unsqueeze(self, dim: int) -> Rotation:
        if dim >= len(self.shape):
            raise ValueError('Invalid dimension')

        rot_mats = self._mat.unsqueeze(dim if dim >= 0 else dim - 2)
        return Rotation(mat=rot_mats)

    def map_tensor_fn(self, fn: Callable[[torch.Tensor],
                                         torch.Tensor]) -> Rotation:
        mat = self._mat.view(self._mat.shape[:-2] + (9, ))
        mat = torch.stack(list(map(fn, torch.unbind(mat, dim=-1))), dim=-1)
        mat = mat.view(mat.shape[:-1] + (3, 3))
        return Rotation(mat=mat)

    @staticmethod
    def cat(rs: Sequence[Rotation], dim: int) -> Rotation:
        rot_mats = [r.rot_mat for r in rs]
        rot_mats = torch.cat(rot_mats, dim=dim if dim >= 0 else dim - 2)

        return Rotation(mat=rot_mats)

    def cuda(self) -> Rotation:
        return Rotation(mat=self._mat.cuda())

    def to(self, device: Optional[torch.device],
           dtype: Optional[torch.dtype]) -> Rotation:
        return Rotation(mat=self._mat.to(device=device, dtype=dtype))

    def type(self, dtype: Optional[torch.dtype]) -> Rotation:
        return Rotation(mat=self._mat.type(dtype))

    def detach(self) -> Rotation:
        return Rotation(mat=self._mat.detach())


class Frame:

    def __init__(
        self,
        rotation: Optional[Rotation],
        translation: Optional[torch.Tensor],
    ):
        if rotation is None and translation is None:
            rotation = Rotation.identity((0, ))
            translation = zero_translation((0, ))
        elif translation is None:
            translation = zero_translation(rotation.shape, rotation.dtype,
                                           rotation.device,
                                           rotation.requires_grad)

        elif rotation is None:
            rotation = Rotation.identity(
                translation.shape[:-1],
                translation.dtype,
                translation.device,
                translation.requires_grad,
            )

        if (rotation.shape != translation.shape[:-1]) or (rotation.device
                                                          !=  # noqa W504
                                                          translation.device):
            raise ValueError('RotationMatrix and translation incompatible')

        self._r = rotation
        self._t = translation

    @staticmethod
    def identity(
        shape: Iterable[int],
        dtype: Optional[torch.dtype] = torch.float,
        device: Optional[torch.device] = torch.device('cpu'),
        requires_grad: bool = False,
    ) -> Frame:
        return Frame(
            Rotation.identity(shape, dtype, device, requires_grad),
            zero_translation(shape, dtype, device, requires_grad),
        )

    def __getitem__(
        self,
        index: Any,
    ) -> Frame:
        if type(index) != tuple:
            index = (index, )

        return Frame(
            self._r[index],
            self._t[index + (slice(None), )],
        )

    def __mul__(
        self,
        right: torch.Tensor,
    ) -> Frame:
        if not (isinstance(right, torch.Tensor)):
            raise TypeError('The other multiplicand must be a Tensor')

        new_rots = self._r * right
        new_trans = self._t * right[..., None]

        return Frame(new_rots, new_trans)

    def __rmul__(
        self,
        left: torch.Tensor,
    ) -> Frame:
        return self.__mul__(left)

    @property
    def shape(self) -> torch.Size:
        s = self._t.shape[:-1]
        return s

    @property
    def device(self) -> torch.device:
        return self._t.device

    def get_rots(self) -> Rotation:
        return self._r

    def get_trans(self) -> torch.Tensor:
        return self._t

    def compose(
        self,
        other: Frame,
    ) -> Frame:
        new_rot = self._r @ other._r
        new_trans = self._r.apply(other._t) + self._t
        return Frame(new_rot, new_trans)

    def apply(
        self,
        pts: torch.Tensor,
    ) -> torch.Tensor:
        rotated = self._r.apply(pts)
        return rotated + self._t

    def invert_apply(self, pts: torch.Tensor) -> torch.Tensor:
        pts = pts - self._t
        return self._r.invert_apply(pts)

    def invert(self) -> Frame:
        rot_inv = self._r.invert()
        trn_inv = rot_inv.apply(self._t)

        return Frame(rot_inv, -1 * trn_inv)

    def map_tensor_fn(self, fn: Callable[[torch.Tensor],
                                         torch.Tensor]) -> Frame:
        new_rots = self._r.map_tensor_fn(fn)
        new_trans = torch.stack(
            list(map(fn, torch.unbind(self._t, dim=-1))), dim=-1)

        return Frame(new_rots, new_trans)

    def to_tensor_4x4(self) -> torch.Tensor:
        tensor = self._t.new_zeros((*self.shape, 4, 4))
        tensor[..., :3, :3] = self._r.rot_mat
        tensor[..., :3, 3] = self._t
        tensor[..., 3, 3] = 1
        return tensor

    @staticmethod
    def from_tensor_4x4(t: torch.Tensor) -> Frame:
        if t.shape[-2:] != (4, 4):
            raise ValueError('Incorrectly shaped input tensor')

        rots = Rotation(mat=t[..., :3, :3])
        trans = t[..., :3, 3]

        return Frame(rots, trans)

    @staticmethod
    def from_3_points(
        p_neg_x_axis: torch.Tensor,
        origin: torch.Tensor,
        p_xy_plane: torch.Tensor,
        eps: float = 1e-8,
    ) -> Frame:
        p_neg_x_axis = torch.unbind(p_neg_x_axis, dim=-1)
        origin = torch.unbind(origin, dim=-1)
        p_xy_plane = torch.unbind(p_xy_plane, dim=-1)

        e0 = [c1 - c2 for c1, c2 in zip(origin, p_neg_x_axis)]
        e1 = [c1 - c2 for c1, c2 in zip(p_xy_plane, origin)]

        denom = torch.sqrt(sum((c * c for c in e0)) + eps)
        e0 = [c / denom for c in e0]
        dot = sum((c1 * c2 for c1, c2 in zip(e0, e1)))
        e1 = [c2 - c1 * dot for c1, c2 in zip(e0, e1)]
        denom = torch.sqrt(sum((c * c for c in e1)) + eps)
        e1 = [c / denom for c in e1]
        e2 = [
            e0[1] * e1[2] - e0[2] * e1[1],
            e0[2] * e1[0] - e0[0] * e1[2],
            e0[0] * e1[1] - e0[1] * e1[0],
        ]

        rots = torch.stack([c for tup in zip(e0, e1, e2) for c in tup], dim=-1)
        rots = rots.reshape(rots.shape[:-1] + (3, 3))

        rot_obj = Rotation(mat=rots)

        return Frame(rot_obj, torch.stack(origin, dim=-1))

    def unsqueeze(
        self,
        dim: int,
    ) -> Frame:
        if dim >= len(self.shape):
            raise ValueError('Invalid dimension')
        rots = self._r.unsqueeze(dim)
        trans = self._t.unsqueeze(dim if dim >= 0 else dim - 1)

        return Frame(rots, trans)

    @staticmethod
    def cat(
        Ts: Sequence[Frame],
        dim: int,
    ) -> Frame:
        rots = Rotation.cat([T._r for T in Ts], dim)
        trans = torch.cat([T._t for T in Ts], dim=dim if dim >= 0 else dim - 1)

        return Frame(rots, trans)

    def apply_rot_fn(self, fn: Callable[[Rotation], Rotation]) -> Frame:
        return Frame(fn(self._r), self._t)

    def apply_trans_fn(self, fn: Callable[[torch.Tensor],
                                          torch.Tensor]) -> Frame:
        return Frame(self._r, fn(self._t))

    def scale_translation(self, trans_scale_factor: float) -> Frame:
        # fn = lambda t: t * trans_scale_factor
        def fn(t):
            return t * trans_scale_factor

        return self.apply_trans_fn(fn)

    def stop_rot_gradient(self) -> Frame:
        # fn = lambda r: r.detach()
        def fn(r):
            return r.detach()

        return self.apply_rot_fn(fn)

    @staticmethod
    def make_transform_from_reference(n_xyz, ca_xyz, c_xyz, eps=1e-20):
        input_dtype = ca_xyz.dtype
        n_xyz = n_xyz.float()
        ca_xyz = ca_xyz.float()
        c_xyz = c_xyz.float()
        n_xyz = n_xyz - ca_xyz
        c_xyz = c_xyz - ca_xyz

        c_x, c_y, d_pair = [c_xyz[..., i] for i in range(3)]
        norm = torch.sqrt(eps + c_x**2 + c_y**2)
        sin_c1 = -c_y / norm
        cos_c1 = c_x / norm

        c1_rots = sin_c1.new_zeros((*sin_c1.shape, 3, 3))
        c1_rots[..., 0, 0] = cos_c1
        c1_rots[..., 0, 1] = -1 * sin_c1
        c1_rots[..., 1, 0] = sin_c1
        c1_rots[..., 1, 1] = cos_c1
        c1_rots[..., 2, 2] = 1

        norm = torch.sqrt(eps + c_x**2 + c_y**2 + d_pair**2)
        sin_c2 = d_pair / norm
        cos_c2 = torch.sqrt(c_x**2 + c_y**2) / norm

        c2_rots = sin_c2.new_zeros((*sin_c2.shape, 3, 3))
        c2_rots[..., 0, 0] = cos_c2
        c2_rots[..., 0, 2] = sin_c2
        c2_rots[..., 1, 1] = 1
        c2_rots[..., 2, 0] = -1 * sin_c2
        c2_rots[..., 2, 2] = cos_c2

        c_rots = Rotation.mat_mul_mat(c2_rots, c1_rots)
        n_xyz = Rotation.mat_mul_vec(c_rots, n_xyz)

        _, n_y, n_z = [n_xyz[..., i] for i in range(3)]
        norm = torch.sqrt(eps + n_y**2 + n_z**2)
        sin_n = -n_z / norm
        cos_n = n_y / norm

        n_rots = sin_c2.new_zeros((*sin_c2.shape, 3, 3))
        n_rots[..., 0, 0] = 1
        n_rots[..., 1, 1] = cos_n
        n_rots[..., 1, 2] = -1 * sin_n
        n_rots[..., 2, 1] = sin_n
        n_rots[..., 2, 2] = cos_n

        rots = Rotation.mat_mul_mat(n_rots, c_rots)

        rots = rots.transpose(-1, -2)
        rot_obj = Rotation(mat=rots.type(input_dtype))

        return Frame(rot_obj, ca_xyz.type(input_dtype))

    def cuda(self) -> Frame:
        return Frame(self._r.cuda(), self._t.cuda())

    @property
    def dtype(self) -> torch.dtype:
        assert self._r.dtype == self._t.dtype
        return self._r.dtype

    def type(self, dtype) -> Frame:
        return Frame(self._r.type(dtype), self._t.type(dtype))


class Quaternion:

    def __init__(self, quaternion: torch.Tensor, translation: torch.Tensor):
        if quaternion.shape[-1] != 4:
            raise ValueError(f'incorrect quaternion shape: {quaternion.shape}')
        self._q = quaternion
        self._t = translation

    @staticmethod
    def identity(
        shape: Iterable[int],
        dtype: Optional[torch.dtype] = torch.float,
        device: Optional[torch.device] = torch.device('cpu'),
        requires_grad: bool = False,
    ) -> Quaternion:
        trans = zero_translation(shape, dtype, device, requires_grad)
        quats = torch.zeros((*shape, 4),
                            dtype=dtype,
                            device=device,
                            requires_grad=requires_grad)
        with torch.no_grad():
            quats[..., 0] = 1
        return Quaternion(quats, trans)

    def get_quats(self):
        return self._q

    def get_trans(self):
        return self._t

    def get_rot_mats(self):
        quats = self.get_quats()
        rot_mats = Quaternion.quat_to_rot(quats)
        return rot_mats

    @staticmethod
    def quat_to_rot(normalized_quat):
        global _QUAT_TO_ROT_tensor
        dtype = normalized_quat.dtype
        normalized_quat = normalized_quat.float()
        if _QUAT_TO_ROT_tensor.device != normalized_quat.device:
            _QUAT_TO_ROT_tensor = _QUAT_TO_ROT_tensor.to(
                normalized_quat.device)
        rot_tensor = torch.sum(
            _QUAT_TO_ROT_tensor * normalized_quat[..., :, None, None]
            * normalized_quat[..., None, :, None],
            dim=(-3, -2),
        )
        rot_tensor = rot_tensor.type(dtype)
        rot_tensor = rot_tensor.view(*rot_tensor.shape[:-1], 3, 3)
        return rot_tensor

    @staticmethod
    def normalize_quat(quats):
        dtype = quats.dtype
        quats = quats.float()
        quats = quats / torch.linalg.norm(quats, dim=-1, keepdim=True)
        quats = quats.type(dtype)
        return quats

    @staticmethod
    def quat_multiply_by_vec(quat, vec):
        dtype = quat.dtype
        quat = quat.float()
        vec = vec.float()
        global _QUAT_MULTIPLY_BY_VEC_tensor
        if _QUAT_MULTIPLY_BY_VEC_tensor.device != quat.device:
            _QUAT_MULTIPLY_BY_VEC_tensor = _QUAT_MULTIPLY_BY_VEC_tensor.to(
                quat.device)
        mat = _QUAT_MULTIPLY_BY_VEC_tensor
        reshaped_mat = mat.view((1, ) * len(quat.shape[:-1]) + mat.shape)
        return torch.sum(
            reshaped_mat * quat[..., :, None, None] * vec[..., None, :, None],
            dim=(-3, -2),
        ).type(dtype)

    def compose_q_update_vec(self,
                             q_update_vec: torch.Tensor,
                             normalize_quats: bool = True) -> torch.Tensor:
        quats = self.get_quats()
        new_quats = quats + Quaternion.quat_multiply_by_vec(
            quats, q_update_vec)
        if normalize_quats:
            new_quats = Quaternion.normalize_quat(new_quats)
        return new_quats

    def compose_update_vec(
        self,
        update_vec: torch.Tensor,
        pre_rot_mat: Rotation,
    ) -> Quaternion:
        q_vec, t_vec = update_vec[..., :3], update_vec[..., 3:]
        new_quats = self.compose_q_update_vec(q_vec)

        trans_update = pre_rot_mat.apply(t_vec)
        new_trans = self._t + trans_update

        return Quaternion(new_quats, new_trans)

    def stop_rot_gradient(self) -> Quaternion:
        return Quaternion(self._q.detach(), self._t)
