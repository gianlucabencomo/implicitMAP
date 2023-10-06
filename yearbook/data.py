import os
import pickle
from typing import Tuple, Any

import jax
import jax.numpy as jnp
import jax.random as jr

from PIL import Image

from torchvision import transforms

"""
setup instructions: 
1. download yearbook dataset at https://people.eecs.berkeley.edu/~shiry/projects/yearbooks/yearbooks.html
2. unzip file and rename folder as data in current path.
"""

FILE = "/yearbook.pkl"
ROOT = "./data"
TRANSFORM = transforms.Compose(
    [
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
    ]
)


class Yearbook:
    def __init__(self, root: str = ROOT, transform: Any = TRANSFORM):
        """Initialize and configure Yearbook dataset for experiments reported in paper.

        Args:
            root (str) : file path for yearbook dataset.
            transform (Any) : transform to apply to yearbook dataset.

        Returns:
            None.

        Raises:
            None.
        """
        self.x_init_years = jnp.array(
            [
                1905,
                1906,
                1908,
                1909,
                1910,
                1911,
                1912,
                1913,
                1914,
                1915,
                1916,
                1919,
                1922,
                1923,
                1924,
                1925,
                1926,
                1927,
                1928,
                1929,
                1930,
            ]
        )
        self.transform = transform
        img_path = []
        years = []
        targets = []
        for f in os.listdir(root + "/F"):
            targets.append(jnp.array(0))
            years.append(int(f[:4]))
            img_path.append(os.path.join(root + "/F", f))
        for f in os.listdir(root + "/M"):
            targets.append(jnp.array(1))
            years.append(int(f[:4]))
            img_path.append(os.path.join(root + "/M", f))
        years = jnp.array(years)
        self.years = [
            year
            for year in jnp.unique(years)
            if (
                year not in self.x_init_years
                and year not in jnp.array([2011, 2012, 2013])
            )
        ]
        if os.path.exists(ROOT + FILE):
            with open(ROOT + FILE, "rb") as file:
                self.data = pickle.load(file)
        else:
            self.data = [
                []
                for year in jnp.unique(years)
                if (
                    year not in self.x_init_years
                    and year not in jnp.array([2011, 2012, 2013])
                )
            ]
            self.data = [[]] + self.data  # for initial
            for i in range(len(years)):
                path = img_path[i]
                sample = Image.open(path).convert("L")
                if self.transform is not None:
                    sample = self.transform(sample)
                sample = jnp.array(sample).reshape(1, 32, 32, 1)
                if years[i] in self.x_init_years:
                    self.data[0].append((sample, targets[i]))
                elif years[i] not in jnp.array([2011, 2012, 2013]):
                    ind = self.years.index(years[i]) + 1
                    self.data[ind].append((sample, targets[i]))
            with open(ROOT + FILE, "wb") as file:
                pickle.dump(self.data, file)

    def get_year(self, year: int) -> Tuple[jax.Array, jax.Array]:
        """Get all of the data (X, y) for a given year and return as jax arrays."""
        X = [i[0] for i in self.data[year]]
        y = [i[1] for i in self.data[year]]
        return jnp.concatenate(X, axis=0), jnp.array(y)

    def load(
        self, key: jr.PRNGKey, n_train: int = 32, n_val: int = 16
    ) -> Tuple[list, list, list, list, jax.Array, jax.Array]:
        X_init, y_init = self.get_year(0)
        X_train, y_train, X_test, y_test = [], [], [], []
        for i in range(1, len(self) + 1):
            X, y = self.get_year(i)
            p = jr.permutation(key, X.shape[0])
            X_train.append(X[p[:n_train]])
            y_train.append(y[p[:n_train]])
            X_test.append(X[p[n_train + n_val : n_train + n_val + 100]])
            y_test.append(y[p[n_train + n_val : n_train + n_val + 100]])
            key, _ = jr.split(key)
        return X_train, y_train, X_test, y_test, X_init, y_init

    def __len__(self) -> int:
        """Return number of years in dataset, including init year."""
        return len(self.years)


if __name__ == "__main__":
    data = Yearbook()
    print(jnp.array(data.years))
    print(len(data.data))
    for i in range(len(data.data)):
        print(f"{1931 + i - 1} : {len(data.data[i])}")
