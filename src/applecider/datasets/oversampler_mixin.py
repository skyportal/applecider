import numpy as np

class OversamplerMixin():
    """Mixin class for handling oversampling in datasets.

    This mixin provides methods to calculate the required number of additional samples
    per class to achieve a desired class distribution, as well as to retrieve the
    appropriate index for oversampled data points.

    Using this mixin requires the child class to implement two methods:
    - _get_class_counts() -> np.ndarray[int]:
        Returns the count of samples per class in the dataset.
    - _get_class_at_index() -> np.ndarray[int]:
        Returns the class index for each data sample in the dataset.

    Calling _calculate_over_sampling_counts() will compute the additional samples
    needed per class based on the ideal class distribution provided. It will introduce
    two new attributes to the class:
    - self.additional_samples_per_class: list[int] - A list indicating how many additional samples
      are needed for each class.
    - self.total_count_with_oversampling: int - The total number of samples in the dataset
      after oversampling is applied.

    """

    def _calculate_over_sampling_counts(self, ideal_class_distribution: list = None, class_counts: list = None):
        """Calculate the number of additional samples needed per class to achieve
        the desired class distribution.
        Parameters
        ----------
        ideal_class_distribution : list
            A list of percentages representing the desired class distribution.
        class_counts : list
            A list of integers representing the current count of samples per class.

        Returns
        -------
        None
            This method sets two attributes:
            - self.additional_samples_per_class: list[int] - A list indicating how
            many additional samples are needed for each class.
            - self.total_count_with_oversampling: int - The total number of samples
            in the dataset after oversampling is applied.
        """

        if ideal_class_distribution is None:
            raise ValueError("ideal_class_distribution must be provided as a numeric list.")
        if class_counts is None:
            raise ValueError("class_counts must be provided as a list of integers.")

        # Cast and normalize the provided ideal class distribution
        unnormalized_percentages = np.array(ideal_class_distribution)
        p_norm = unnormalized_percentages / np.sum(unnormalized_percentages)

        # Calculate the total current count
        total_current = np.sum(class_counts)

        req_totals = np.zeros_like(p_norm, dtype=np.int64)
        nonzero_mask = p_norm > 0
        req_totals[nonzero_mask] = np.ceil(class_counts[nonzero_mask] / p_norm[nonzero_mask]).astype(np.int64)

        # minimal feasible total is the maximum of those and at least current total
        minimal_total = max(int(req_totals.max()), int(total_current))

        # build integer target counts for minimal_total using floor + allocate remainder by fractional parts
        target_real = p_norm * minimal_total
        target_floor = np.floor(target_real).astype(np.int64)
        remainder = minimal_total - target_floor.sum()
        if remainder > 0:
            residuals = target_real - target_floor
            # indices sorted by residual descending
            order = np.argsort(residuals)[::-1]
            for idx in order[:remainder]:
                target_floor[idx] += 1

        self.additional_samples_per_class = target_floor - class_counts
        self.total_count_with_oversampling = np.sum(target_floor)


    def prepare_over_sampling(self, ideal_class_distribution: list = None, class_at_index: list = None):
        """Calculate the oversampling indices for the dataset based on the ideal class distribution.

        Parameters
        ----------
        ideal_class_distribution : list
            A list of percentages representing the desired class distribution.
        class_at_index : list
            A list of integers representing the class index for each data sample.

        Returns
        -------
        None
            This method sets the attribute:
            - self.class_at_index: np.ndarray[int] - A numpy array where each integer
            corresponds to the class index of the data sample at that index.
        """

        if ideal_class_distribution is None:
            raise ValueError("ideal_class_distribution must be provided as a numeric list.")
        if class_at_index is None:
            raise ValueError("class_at_index must be provided as a list of integers.")
        rng = np.random.default_rng()
        self._class_at_index = np.array(class_at_index)
        self._original_count = len(self._class_at_index)

        # get class counts from list of class at index
        _, class_counts = np.unique(self._class_at_index, return_counts=True)

        # Calculate the additional oversampling counts
        self._calculate_over_sampling_counts(ideal_class_distribution, class_counts)

        # Create an array that will map oversampled indices back to original indices
        oversampled_idx_to_original_idx = np.stack([np.arange(self._original_count), np.zeros(self._original_count, dtype=int)])

        # Fill the oversampled_to_original_index array
        for class_index, additional_count in enumerate(self.additional_samples_per_class):
            original_indices_for_class = np.where(self._class_at_index == class_index)[0]
            # Randomly select N original indexes from the current class
            selected_indexes = rng.choice(original_indices_for_class, size=additional_count, replace=True)
            # stack selected_indexes on an array of ones to represent oversampled entries
            selected_indexes = np.stack([selected_indexes, np.ones(len(selected_indexes), dtype=int)])
            #concatenate original_index to oversampled_idx_to_original_idx
            oversampled_idx_to_original_idx = np.hstack((oversampled_idx_to_original_idx, selected_indexes))

        # Shuffle all the indexes (including the original indexes) so that batches
        # drawn from the array will have a consistent distribution
        rng.shuffle(oversampled_idx_to_original_idx, axis=1)

        self._oversampled_idx_to_original_idx = oversampled_idx_to_original_idx

    def retrieve_oversampled_index(self, index: int) -> tuple[int, bool]:
        """
        This function will return a specific index from the original dataset as well
        as a boolean indicating if the index corresponds to an oversampled instance.

        The boolean can then be used to determine if any data augmentation or jittering
        should be applied to the original data sample.

        Parameters
        ----------
        index : int
            The index to retrieve from the over-sampled dataset.

        Returns
        -------
        tuple
            The returned tuple containing (original_index : int, is_oversampled : bool)

        """
        returned_index = self._oversampled_idx_to_original_idx[0, index]
        is_oversampled = self._oversampled_idx_to_original_idx[1, index] == 1

        return returned_index, is_oversampled
