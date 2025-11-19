import numpy as np

class Oversampler():

    def __init__(self):
        self.total_count_with_oversampling = 0

    def _get_class_counts(self) -> np.ndarray[int]:
        """Method to be implemented by the child class that will return a list of
        integers representing the total number of data samples per class.

        Returns
        -------
        list[int]
            A list of integers where each integer corresponds to the count of samples
            in each class.

        Raises
        ------
        NotImplementedError
            This must be implemented by the child class.
        """
        raise NotImplementedError

    def _get_class_at_index(self) -> np.ndarray[int]:
        """Method to be implemented by the child class that will return a list of
        integers representing the class index for each data sample in the dataset.

        Returns
        -------
        np.ndarray[int]
            A numpy array of integers where each integer corresponds to the class index
            of the data sample at that index.

        Raises
        ------
        NotImplementedError
            This must be implemented by the child class.
        """
        raise NotImplementedError

    def _calculate_over_sampling_counts(self, ideal_class_distribution: list = None):
        unnormalized_percentages = np.array(ideal_class_distribution)
        p_norm = unnormalized_percentages / np.sum(unnormalized_percentages)

        class_counts = self._get_class_counts()

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

        target_counts = target_floor
        self._oversample_additions = target_counts - class_counts
        self.total_count_with_oversampling = np.sum(target_counts)

    def retrieve_index(self, index):
        """
        This function will determine if the provided index corresponds to an original
        or over-sampled instance. If it's original, it returns the index as is. If
        it's over-sampled, it randomly selects an index from the original dataset
        that belongs to the same class as the over-sampled instance.

        Args:
            index (int): The index in the over-sampled dataset.
        Returns:
            tuple: (original_index (int), is_oversampled (bool))
        """
        # Determine which class the given index corresponds to in the over-sampled dataset
        original_count = np.sum(self._get_class_counts())
        if index < original_count:
            return index, False
        else:
            over_sampled_index = index - original_count
            # calculate the cumulative sum of the additions, find the index of the class
            cumulative_additions = np.cumsum(self._oversample_additions)
            # find the class index where the over_sampled_index would fit
            class_index = np.searchsorted(cumulative_additions, over_sampled_index, side='left')
            # randomly select an index from self.class_at_index where class matches class_index
            class_indices = np.where(self._get_class_at_index() == class_index)[0]
            random_choice = np.random.choice(class_indices)
            return random_choice, True