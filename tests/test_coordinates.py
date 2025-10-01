import jax; jax.config.update("jax_enable_x64", True)  # noqa: E702
import jax.numpy as jnp

from temgym_core import PixelsYX, PixelYX, CoordsXY, CoordXY


def test_to_pixels_int():
    px = PixelYX(x=17, y=23)
    pxs = px.to_pixels()
    assert isinstance(pxs, PixelsYX)
    assert len(pxs.x) == 1
    assert len(pxs.y) == 1
    assert jnp.all(px.x == pxs.x)
    assert jnp.all(px.y == pxs.y)
    assert pxs.x.dtype.kind == 'i'
    assert pxs.y.dtype.kind == 'i'


def test_to_pixels_float():
    px = PixelYX(x=17., y=23.)
    pxs = px.to_pixels()
    assert isinstance(pxs, PixelsYX)
    assert len(pxs.x) == 1
    assert len(pxs.y) == 1
    assert jnp.all(px.x == pxs.x)
    assert jnp.all(px.y == pxs.y)
    assert pxs.x.dtype.kind == 'f'
    assert pxs.y.dtype.kind == 'f'


def test_to_coords():
    coord = CoordXY(x=17., y=23.)
    coords = coord.to_coords()
    assert isinstance(coords, CoordsXY)
    assert len(coords.x) == 1
    assert len(coords.y) == 1
    assert jnp.all(coord.x == coords.x)
    assert jnp.all(coord.y == coords.y)
    assert coords.x.dtype.kind == 'f'
    assert coords.y.dtype.kind == 'f'
