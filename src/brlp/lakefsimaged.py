import os
import numpy as np
from monai.data.image_reader import ImageReader
from pathlib import Path

from monai.transforms import LoadImaged
from monai.config import DtypeLike, KeysCollection
from monai.utils import ensure_tuple
from monai.utils.enums import PostFix

from lakefsloader import LakeFSLoader

DEFAULT_POST_FIX = PostFix.meta()

class LakeFSCacheError(Exception):
    """
    Raise this error in the special case when you are able to download files into the cache from lakefs, but the file disappears.
    """

class LoadLakeFSImaged(LoadImaged):
    """
    Wrapper of `LoadImaged`.

    Additional Args:
        lakefs_loader (LakeFSLoader): the lakefs loader to use, with defined the endpoint and keys.
                                    Acts like the monai.transforms.LoadImaged if not given.
        max_attempts (int): how many times to try and redownload the file. To handle if the file gets deleted while accesing it.
    """
    def __init__(
            self,
            keys: KeysCollection,
            lakefs_loader: LakeFSLoader = None,
            max_attempts: int = 10, 
            reader: type[ImageReader] | str | None = None,
            dtype: DtypeLike = np.float32, 
            meta_keys: KeysCollection | None = None, 
            meta_key_postfix: str = DEFAULT_POST_FIX, 
            overwriting: bool = False, 
            image_only: bool = True, 
            ensure_channel_first: bool = False, 
            simple_keys: bool = False, 
            prune_meta_pattern: str | None = None, 
            prune_meta_sep: str = ".", 
            allow_missing_keys: bool = False, 
            expanduser: bool = True, 
            *args, 
            **kwargs
        ) -> None:
        super().__init__(keys, reader, dtype, meta_keys, meta_key_postfix, overwriting, image_only, ensure_channel_first,
                        simple_keys, prune_meta_pattern, prune_meta_sep, allow_missing_keys, expanduser, *args, **kwargs)
        
        self.lakefs_loader = lakefs_loader
        self.max_attempts = max_attempts

    def _ensure_s3d(self, data):
        """
        Iterates through the given keys and ensures for each that all files is available.
        Depends on the file suffix: if none is given, it is assumed to be a dicom directory and recursively downloaded.
        """
        d = dict(data)
        for key in self.key_iterator(d):
            
            filenames = tuple(f"{Path(s).expanduser()}" for s in ensure_tuple(d[key]))
            for file in filenames:
                _, extension = os.path.splitext(file)
                if extension:
                    self.lakefs_loader.check_file(file)
                else:
                    self.lakefs_loader.check_dir(file)

        return

    def __call__(self, data, reader: ImageReader | None = None):
        """
        Calls LoadImaged, handles FileNotFound errors by downloading the missing file(s) from the S3 storage if possible.
        """
        if self.lakefs_loader:
            for _ in range(self.max_attempts):
                try:
                    return super().__call__(data, reader)

                except FileNotFoundError:
                    self._ensure_s3d(data)

            raise LakeFSCacheError(f"Tried {self.max_attempts} times to access the files from {data}. Downloading the files from Lakefs \
                                    has worked, but they seem to be deleted before reading them. Check the cache hanlding.")
        else:
            return super().__call__(data, reader)