""" Embedding Strategies """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import time
from typing import List, Union, Tuple

import numpy as np
import torch
import lightly
from lightly.embedding._base import BaseEmbedding
from tqdm import tqdm

from lightly.utils.reordering import sort_items_by_keys

if lightly._is_prefetch_generator_available():
    from prefetch_generator import BackgroundGenerator


import os
import stat

_fd_types = (
    ('REG', stat.S_ISREG),
    ('FIFO', stat.S_ISFIFO),
    ('DIR', stat.S_ISDIR),
    ('CHR', stat.S_ISCHR),
    ('BLK', stat.S_ISBLK),
    ('LNK', stat.S_ISLNK),
    ('SOCK', stat.S_ISSOCK)
)

def fd_table_status():
    result = []
    for fd in range(100):
        try:
            s = os.fstat(fd)
        except:
            continue
        for fd_type, func in _fd_types:
            if func(s.st_mode):
                break
        else:
            fd_type = str(s.st_mode)
        result.append((fd, fd_type))
    return result

def fd_table_status_logify(fd_table_result):
    return ('Open file handles: ' +
            ', '.join(['{0}: {1}'.format(*i) for i in fd_table_result]))

def fd_table_status_str():
    return fd_table_status_logify(fd_table_status())

def fd_table_count():
    return len(fd_table_status())


class SelfSupervisedEmbedding(BaseEmbedding):
    """Implementation of self-supervised embedding models.

    Implements an embedding strategy based on self-supervised learning. A
    model backbone, self-supervised criterion, optimizer, and dataloader are
    passed to the constructor. The embedding itself is a pytorch-lightning
    module.

    The implementation is based on contrastive learning.

    * SimCLR: https://arxiv.org/abs/2002.05709
    * MoCo: https://arxiv.org/abs/1911.05722
    * SimSiam: https://arxiv.org/abs/2011.10566

    Attributes:
        model:
            A backbone convolutional network with a projection head.
        criterion:
            A contrastive loss function.
        optimizer:
            A PyTorch optimizer.
        dataloader:
            A torchvision dataloader.
        scheduler:
            A PyTorch learning rate scheduler.

    Examples:
        >>> # define a model, criterion, optimizer, and dataloader above
        >>> import lightly.embedding as embedding
        >>> encoder = SelfSupervisedEmbedding(
        >>>     model,
        >>>     criterion,
        >>>     optimizer,
        >>>     dataloader,
        >>> )
        >>> # train the self-supervised embedding with default settings
        >>> encoder.train_embedding()
        >>> # pass pytorch-lightning trainer arguments as kwargs
        >>> encoder.train_embedding(max_epochs=10)

    """

    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        dataloader: torch.utils.data.DataLoader,
        scheduler=None,
    ):

        super(SelfSupervisedEmbedding, self).__init__(
            model, criterion, optimizer, dataloader, scheduler
        )

    def embed(self,
              dataloader: torch.utils.data.DataLoader,
              device: torch.device = None
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Embeds images in a vector space.

        Args:
            dataloader:
                A PyTorch dataloader.
            device:
                Selected device (`cpu`, `cuda`, see PyTorch documentation)

        Returns:
            Tuple of (embeddings, labels, filenames) ordered by the
            samples in the dataset of the dataloader.
                embeddings:
                    Embedding of shape (n_samples, embedding_feature_size).
                    One embedding for each sample.
                labels:
                    Labels of shape (n_samples, ).
                filenames:
                    The filenames from dataloader.dataset.get_filenames().


        Examples:
            >>> # embed images in vector space
            >>> embeddings, labels, fnames = encoder.embed(dataloader)

        """

        print('before', fd_table_count())

        self.model.eval()
        embeddings, labels, fnames = None, None, []

        if lightly._is_prefetch_generator_available():
            pbar = tqdm(
                BackgroundGenerator(dataloader, max_prefetch=3),
                total=len(dataloader)
            )
        else:
            pbar = tqdm(dataloader, total=len(dataloader))

        print('tqdm', fd_table_count())
        efficiency = 0.0
        embeddings = []
        labels = []
        with torch.no_grad():

            start_timepoint = time.time()
            for (img, label, fname) in pbar:

                print(f'{label} ', fd_table_count())

                # this following 2 lines are needed to prevent a file handler leak,
                # see https://github.com/lightly-ai/lightly/pull/676
                img = img.to(device)
                label = label.clone()

                fnames += [*fname]

                batch_size = img.shape[0]
                prepared_timepoint = time.time()

                emb = self.model.backbone(img)
                emb = emb.detach().reshape(batch_size, -1)

                embeddings.append(emb)
                labels.append(label)

                finished_timepoint = time.time()

                data_loading_time = prepared_timepoint - start_timepoint
                inference_time = finished_timepoint - prepared_timepoint
                total_batch_time = data_loading_time + inference_time

                efficiency = inference_time / total_batch_time
                pbar.set_description("Compute efficiency: {:.2f}".format(efficiency))
                start_timepoint = time.time()

            print('after', fd_table_count())
            embeddings = torch.cat(embeddings, 0)
            labels = torch.cat(labels, 0)

            embeddings = embeddings.cpu().numpy()
            labels = labels.cpu().numpy()

        print('before sort', fd_table_count())
        sorted_filenames = dataloader.dataset.get_filenames()
        sorted_embeddings = sort_items_by_keys(
            fnames, embeddings, sorted_filenames
        )
        sorted_labels = sort_items_by_keys(
            fnames, labels, sorted_filenames
        )
        embeddings = np.stack(sorted_embeddings)
        labels = np.stack(sorted_labels)

        print('sorted', fd_table_count())
        return embeddings, labels, sorted_filenames
