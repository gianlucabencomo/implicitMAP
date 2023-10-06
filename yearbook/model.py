import jax
import jax.numpy as jnp
from flax import linen as nn  # Linen API
from jax._src.config import config

config.update("jax_enable_x64", True)


class CNN(nn.Module):
    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass for CNN used in yearbook experiments.

        Args:
            x (jax.Array) : input data of the dimensions N x D x D x 1.

        Returns:
            Result of forward pass.

        Raises:
            None.
        """
        # Layer 1
        x = nn.Conv(features=32, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        # Layer 2
        x = nn.Conv(features=32, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        # # Layer 3
        x = nn.Conv(features=32, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        # # Layer 4
        x = nn.Conv(features=32, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        # Flatten
        x = x.reshape(x.shape[:-3] + (-1,))  # flatten
        x = nn.Dense(features=1)(x)
        return x


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    in_dim = (1, 32, 32, 1)
    cnn = CNN()
    params = cnn.init(key, jnp.zeros(in_dim))["params"]
    cnn.apply({"params": params}, jnp.ones(in_dim))
    print(cnn.tabulate(key, jnp.empty(in_dim)))
