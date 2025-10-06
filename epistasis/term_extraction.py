import collections
import itertools
import pickle
import numpy as np
from typing import Dict, List, Tuple, Iterable
from importlib import reload
from tqdm import tqdm

class TermExtractor(object):
    def __init__(self, cardinalities:Iterable[int], global_max_term_order:int=None):
        """
        
        Args:
            cardinalities (iterable): Iterable containing the cardinalities
                of each component.
            global_max_term_order (int or None): The maximal term order. 
                If this is None, the maximal term order will be set to the number 
                of components.
                (Default: None)
        
        """
        self.cardinalities = tuple(cardinalities)

        # The number of components is given by the number of elements 
        # of self.cardinalities
        self.num_components = len(self.cardinalities)
            
        # Parse the global max term order
        self.global_max_term_order = self._parse_max_term_order('global_max_term_order', global_max_term_order, upper_limit=self.num_components)

        # Generate the term labels dictionary
        self.term_labels_dict = self._generate_term_labels_dict()

        # Initialize to be determined class attributes to None
        self.marginals_max_term_order = None 
        self.marginals_dict           = None
        self.factors_max_term_order   = None 
        self.factors_dict             = None

    @property
    def num_states(self):
        """ Return the number of states given the cardinalities. """
        return int(np.prod(self.cardinalities))

    @classmethod
    def from_file(cls, file_path:str):
        """
        Initialize a term extractor by loading its content from 
        a file containing a previously saved instance.

        Args:
            file_path (str): File path where the content should be stored in.

        Return:
            (TermExtractor): A TermExtractor instance containing
                the loaded content.

        """
        # Open the file in binary mode
        with open(file_path, 'rb') as file:
            # Deserialize and retrieve the variable from the file
            loaded_dict = pickle.load(file)

        # Initialize class instance with loaded quantities
        class_instance = cls(cardinalities=loaded_dict['cardinalities'], global_max_term_order=loaded_dict['global_max_term_order'])

        # Load some of the class attributes with loaded quantities
        class_instance.marginals_dict = loaded_dict['marginals_dict']
        class_instance.factors_dict   = loaded_dict['factors_dict']

        # Determine and class attributes
        # Remark: class_instance.marginals_dict and class_instance.factors_dict contain the
        #         term orders as dictionary-keys.
        class_instance.marginals_max_term_order = max(list(class_instance.marginals_dict.keys()))
        class_instance.factors_max_term_order   = max(list(class_instance.factors_dict.keys()))

        print(f"Loaded content from the file: {file_path}")

        return class_instance


    def to_file(self, file_path:str)->None:
        """
        Save the content of the term extractor in a file.

        Args:
            file_path (str): File path where the content should be stored in.

        Return:
            None
        """
        # Generate a dictionary containing all the content to be saved
        save_dict = dict()
        save_dict['cardinalities']         = self.cardinalities
        save_dict['global_max_term_order'] = self.global_max_term_order
        save_dict['marginals_dict']        = self.marginals_dict
        save_dict['factors_dict']          = self.factors_dict

        # Open the file in binary mode
        with open(file_path, 'wb') as file:
            # Serialize and write the save dictionary to the file
            pickle.dump(save_dict, file)

        print(f"Saved content to the file: {file_path}")

    def determine_marginals(self, state_fitness_dict, marginals_max_term_order:int=None)->None:
        """
        Determine the marginals up to a certain order.

        Args:
            state_fitness_dict (dict): Dictionary containing the states as 
                dictionary-keys and their associated fitness as dictionary-values.
            marginals_max_term_order (int or None): The order up to which
                marginals should be determined to. If None, then
                marginals are determined up to the maximal term order.
                (Default: None)

        Return:
            None

        Remark:
            This is a wrapper and any of the two wrapped methods will set the class 
            attribute self.marginals_dict.            

        """
        # Parse the maximal marginal term order and assign it to the corresponding class attribute
        self.marginals_max_term_order = self._parse_max_term_order('marginals_max_term_order', marginals_max_term_order, upper_limit=self.global_max_term_order)
 
        if len(state_fitness_dict.keys())==self.num_states:
            print(f"The number of states in the 'state_fitness_dict' corresponds to the total number of states, thus determine marginals over complete space.")
            return self._determine_marginals_complete_space(state_fitness_dict)
        else:
            print(f"The number of states in the 'state_fitness_dict' does not correspond to the total number of states, thus determine marginals over incomplemt space.")
            return self._determine_marginals_incomplete_space(state_fitness_dict)

    def _determine_marginals_complete_space(self, state_fitness_dict)->None:
        """
        Determine the marginals up to a certain order.

        Args:
            state_fitness_dict (dict): Dictionary containing the state as 
                dictionary-keys and their associated fitness as dictionary-values.

        Return:
            None

        Remark:
            This method will set the class attribute self.marginals_dict.            

        """

        # Loop over the states and construct the fitness tensor
        # Example:
        # Consider self.cardinalities=[2, 3], so there are 6 states
        # {[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]}.
        # The fitness tensor is then given as tensor of shape (2, 3) of the form
        # fitness_tensor = [
        #                   [fitness_fn([0, 0]), fitness_fn([0, 1]), fitness_fn([0, 2])],
        #                   [fitness_fn([1, 0]), fitness_fn([1, 1]), fitness_fn([1, 2])],
        #                  ]
        fitness_tensor = np.zeros(self.cardinalities)
        #for state, fitness in state_fitness_dict.items():
        print('_determine_marginals_complete_space: loop over states')
        for state, fitness in tqdm(state_fitness_dict.items()):
            # Determine a numpy-slicing-compatible index that corresponds
            # to the current state
            # Example: For a state [0, 1], the corresponding numpy-slicing-compatible
            #          index would be (array(0), array(1)).
            ix = self._get_numpy_array_index(state)

            # Add the fitness to the corresponding entry in the fitness tensor
            fitness_tensor[ix] = fitness

        # Loop over all terms (outer loop over term orders and inner loop over 
        # the term labels of each term order) and determine their marginals
        self.marginals_dict = collections.defaultdict(dict)
        #for term_order in self.term_labels_dict:
        print('_determine_marginals_complete_space: loop over terms')
        for term_order in tqdm(self.term_labels_dict):
            # If the current term order is larger than the required maximal
            # order of the to be determined marginals, skip all terms with
            # this term order and continue to the next term order
            if self.marginals_max_term_order<term_order:
                continue

            # Loop over the term labels with the current term order
            for term_label in self.term_labels_dict[term_order]:
                # Determine the complement components of the current term
                term_complement_components = self._get_term_complement_components(term_label)
                
                # Apply the mean over all the complement components of the current term.
                # Remark: Due to the way the fitness tensor is constructed, the components 
                #         correspond to the axes of the fitness tensor.
                # Example: Consider self.num_components=4 and the term label (1, 3).
                #          This term contains the component 1 and 3, and the complement components 
                #          are 2 and 4. Interpreting the components as axes of the fitness tensor,
                #          one has to take the mean over the axes 2 and 4 in the fitness tensor
                #          to obtain the marginal corresponding to term label (1, 3).
                self.marginals_dict[term_order][term_label] = np.apply_over_axes(np.mean, fitness_tensor, term_complement_components).squeeze()

        # Cast the marginals dictionary to a regular dictionary
        self.marginals_dict = dict(self.marginals_dict)

    def _determine_marginals_incomplete_space(self, state_fitness_dict):        
        """
        Determine the marginals up to a certain order.

        Args:
            state_fitness_dict (dict): Dictionary containing the states as 
                dictionary-keys and their associated fitness as dictionary-values.

        Return:
            None

        Remark:
            This method will set the class attribute self.marginals_dict.            

        """
        fitness_sum_dict = collections.defaultdict(dict)
        counts_dict      = collections.defaultdict(dict)

        # Loop over all states
        #for state, fitness in state_fitness_dict.items():
        print('_determine_marginals_incomplete_space: loop over all states')
        for state, fitness in tqdm(state_fitness_dict.items()):
            # Loop over all terms (outer loop over term orders and inner loop over 
            # the term labels of each term order) and determine their marginals
            self.marginals_dict = collections.defaultdict(dict)
            
            for term_order in self.term_labels_dict:
                # If the current term order is larger than the required maximal
                # order of the to be determined marginals, skip all terms with
                # this term order and continue to the next term order
                if self.marginals_max_term_order<term_order:
                    continue
                # Loop over the term labels with the current term order
                for term_label in self.term_labels_dict[term_order]:
                    if term_order==0:
                        # Initialize dictionary-values to 0 for a new term
                        if term_label not in fitness_sum_dict[term_order]:
                            fitness_sum_dict[term_order][term_label] = 0
                            counts_dict[term_order][term_label]      = 0
                
                        # Add the fitness value
                        fitness_sum_dict[term_order][term_label] += fitness
                        counts_dict[term_order][term_label]      += 1
                    else:
                        if term_label not in fitness_sum_dict[term_order]:
                            # Get the shape of the term values 
                            term_values_shape = self._get_term_values_shape(term_label)
                        
                            # Initialize zero arrays
                            fitness_sum_dict[term_order][term_label] = np.zeros(term_values_shape)
                            counts_dict[term_order][term_label]      = np.zeros(term_values_shape)
                
                        # Determine the numpy-array-slicing compatible indices for the
                        # components in the current term
                        component_states = [state[component] for component in term_label]
                        ix = self._get_numpy_array_index(component_states)
                        fitness_sum_dict[term_order][term_label][ix] += fitness
                        counts_dict[term_order][term_label][ix]      += 1

        # Determine the marginals by normalizing the sums over the 
        # fitness values by the number of added fitness values 
        # (i.e. the counts) for all marginals
        self.marginals_dict = collections.defaultdict(dict)
        #for term_order in fitness_sum_dict.keys():
        print('_determine_marginals_incomplete_space: computing marginals')
        for term_order in tqdm(fitness_sum_dict.keys()):
            for term_label in fitness_sum_dict[term_order]:
                fitness_sum = fitness_sum_dict[term_order][term_label]
                counts      = counts_dict[term_order][term_label]
                if term_order==0:
                    if counts==0:
                        self.marginals_dict[term_order][term_label] = 0
                    else:
                        self.marginals_dict[term_order][term_label] = np.array(fitness_sum/counts)
                else:
                    marginal_values = np.zeros_like(fitness_sum)
                    ix = np.where(0<counts)
                    marginal_values[ix] = fitness_sum[ix]/counts[ix]
                    self.marginals_dict[term_order][term_label] = marginal_values
            
        # Cast the marginals dictionary to a regular dictionary
        self.marginals_dict = dict(self.marginals_dict)

    def determine_factors(self, factors_max_term_order:int=None)->None:
        """
        Determine the factors up to a certain order.

        Args:
            factors_max_term_order (int or None): The order up to which
                factors should be determined to. If None, then factors 
                are determined up to the maximal term order of the marginals.
                (Default: None)

        Return:
            None

        Remark:
            This method will set the class attribute self.factors_dict.            

        """
        # Check that the marginals dictionary has been determined (i.e. is not None any longer)
        if self.marginals_dict is None:
            err_msg = "Cannot determine the factors if the marginals have not been determined yet. Please call the method 'determine_marginals' first."
            raise ValueError(err_msg)

        # Parse the maximal factors term order and assign it to the corresponding class attribute
        self.factors_max_term_order = self.global_max_term_order = self._parse_max_term_order('factors_max_term_order', factors_max_term_order, upper_limit=self.marginals_max_term_order)

        # Get the sorted term orders of the marginals that correspond
        # to the sorted keys of self.marginals_dict)
        term_orders = list(self.marginals_dict.keys())
        term_orders.sort()

        self.factors_dict = collections.defaultdict(dict)
        #for term_order in term_orders:
        print('determine_factors: computing factor values') 
        for term_order in tqdm(term_orders):
            # If the current term order is larger than the required maximal
            # order of the to be determined factors, skip all terms with
            # this term order and continue to the next term order
            if self.factors_max_term_order<term_order:
                continue

            # If the term order is 0, the factor corresponds to the marginal
            if term_order==0:
                self.factors_dict[term_order] = self.marginals_dict[term_order]
            else:
                # For all terms larger than 0, we have to construct the 
                for term_label, marginal in self.marginals_dict[term_order].items():
                    # Determine the sum over the parent term factors
                    sum_parent_term_factors = self._get_sum_parent_term_factors(term_label, marginal.shape)

                    # Determine the factor values given by
                    # f(term) = f^{M}(term) - sum_{parent_term in parents(term)} f(parent_term)
                    self.factors_dict[term_order][term_label] = marginal - sum_parent_term_factors

        # Subtract the mean from the values of all factors (except for the zero-order term) 
        # so that they sum to zero
        #for term_order in self.factors_dict.keys():
        print('determine_factors: zero-sum factor values') 
        for term_order in tqdm(self.factors_dict.keys()):
            if 0<term_order:
                self.factors_dict[term_order] = {term_label: factor-np.mean(factor) for term_label, factor in self.factors_dict[term_order].items()}
        
        # Cast the factors dictionary to a regular dictionary
        self.factors_dict = dict(self.factors_dict)

    def _get_sum_parent_term_factors(self, term_label:Tuple[int], factor_shape)->np.array:
        """
        Return the sum over all factors of the parent terms of one term.

        Args:
            term_label (tuple of ints): Term label.

        Return:
            (np.array) The sum over the term factors as numpy array of the
                shape of the term.

        Implementation details:
            Assuming the term has order term_order, split this term X into a single-order parent x_j
            and a (term_order-1)-term parent term X\\x_j obtained from the term by 'deleting' x_j.
            Look up the marginal of X\\x_j, f^{M}(X\\x_j), and determine the sum of the factors of the
            parents of X involving x_j, F(X; x_j).
            The sum of the factors of ALL parents of X, F(X) are then given by
            F(X) = f^{M}(X\\x_j) + F(X; x_j).
            See example below for an illustration.
            
            Which single-order term x_j is selected from X is arbitrary and we choose the first 
            single-order term appearing in X.

        
        Example:
            Consider the three components {x_1, x_2, x_3}.
            The sum over the factors of parent terms of the term (x_1, x_2, x_3) are given by
            F(x_1, x_2, x_3) = f_0 + f(x_1) + f(x_2) + f_(x_3) + f(x_1, f_x_2) + f(x_1, x_3) + f(x_2, x_3)
            This sum can be rewritten using the marginal of (x_2, x_3) given by
            f^{M}(x_2, x_3) = f_0 + f(x_2) + f_(x_3) + f(x_2, x_3)
            as
            F(x_1, x_2, x_3) = f^{M}(x_2, x_3) + f_(x_1) + f(x_1, x_2) + f(x_1, x_3)
                             = f^{M}(x_2, x_3) + F((x_1, x_2, x_3); x_1)
            where
            F((x_1, x_2, x_3); x_1) = f_(x_1) + f(x_1, x_2) + f(x_1, x_3)
            itself is the sum over all parent terms of (x_1, x_2, x_3) that include x_1.
            The choice of splitting x_1 away from (x_1, x_2, x_3) is arbitrary and one could also
            split x_2 or x_3 and obtain a corresponding term.
        """
        # Split the term into a parent single-order term (choose the first single-order term for this purpose) 
        # and a parent term involving all but this chosen single-order term (i.e. the 'complement' parent term
        # of the single-order term in the entire term).
        # Example: term_label = (2, 4, 5, 7) -> {(2,), (4, 5, 7)}
        single_order_parent_term_label = term_label[0]
        complement_parent_term_label   = term_label[1:]

        # Determine the marginal of the complement parent term and use it to initialize the sum of parent term
        # factors (to which additional factors will be added below).
        sum_parent_term_factors = self._get_parent_term_values('marginal', term_label, complement_parent_term_label)

        # Get all parent terms of the term that include the single-order parent term and sum over them.
        parent_term_labels = self._get_parent_term_labels(term_label, included_term_label=single_order_parent_term_label)
        for parent_term_label in parent_term_labels:
            sum_parent_term_factors += self._get_parent_term_values('factor', term_label, parent_term_label)

        return sum_parent_term_factors
    
    def _get_parent_term_labels(self, term_label:Tuple[int], included_term_label:Tuple[int]=None)->List:
        """
        Return a list of all labels of the parents of a term.

        Args:
            term_label (tuple of ints): Term of which the parent terms should be returned of.
            included_term_label (None or tuple of ints): If None, the method will return all 
                parent. If not None, the method will return only the parent terms including 
                this specific term.

        Return:
            (list of tuples): List of the requested parent term labels (tuples of integers).
        
        Example:
            Consider the term label (1, 2, 3), the parent term labels of this term are
            {(,), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3)} that would be returned if
            included_term_label is None.
            If included_term_label=(2,), only the parent term labels including this term
            would be returned, i.e. {(2,), (1, 2), (2, 3)}.
        """
        # Generate a list of all parent terms (and the term itself)
        parent_term_labels_list = list(self._get_powerset(term_label))

        # Remove the term itself (as it is not a parent of itself)
        parent_term_labels_list.remove(term_label)

        # If a certain term must be included in the parent terms, only keep 
        # the parent terms involving this special to-be-included term
        if included_term_label is not None:
            parent_term_labels_list = list(filter(lambda x: (included_term_label in x), parent_term_labels_list))

        return parent_term_labels_list
    
    def _get_parent_term_values(self, which_values, term_label:Tuple[int], parent_term_label:Tuple[int])->np.array:
        """
        Return the values (marginal or factor) of a parent term in the 'value-space'
        of the term itself, where all the values are repeated along the axes not
        corresponding to the parent term (i.e. along the parent term's complement).

        Args:
            which_values (str): Either 'marginal' or 'factor' depending which
                values should be returned for the term and parent term.
            term_label (tuple of ints): Term.
            parent_term_label (tuple of ints): Specific parent term of the term.
        
        Return:
            (np.array) The parent term values in the 'value-space' of the term.
        """
        # Determine the parent term order
        parent_term_order = len(parent_term_label)

        # Access the values of the parent term, differing the cases where either the
        # marginal or factor values are requested
        if which_values=='marginal':
            parent_term_values = self.marginals_dict[parent_term_order][parent_term_label]
        elif which_values=='factor':
            parent_term_values = self.factors_dict[parent_term_order][parent_term_label]
        else:
            err_msg = f"Input 'which_values' must be either 'marginal' or 'factor', got '{which_values}' instead."
            raise ValueError(err_msg)
        
        return self._repeat_parent_term_values(term_label, parent_term_label, parent_term_values)
    
    def _repeat_parent_term_values(self, term_label, parent_term_label, parent_term_values):
        """
        Repeat the values (marginal or factor) of a parent term along all the axes of the
        other components appearing in the term (i.e. the complement of the parent term).

        Args:
            term_label (tuple of ints): Term.
            parent_term_label (tuple of ints): Parent term of the term.
            parent_term_values (np.array): Values (marginal or factor) of the parent term
                that should be repeated along the complemnt of the parent term (i.e. along
                the axes not corresponding to the parent term) to obtain the parent term
                values in the 'value-space' of the term.
        Return:
            (np.array) Repeated parent term values in the 'value-spaces' of the term.

        Example:
            Consider self.cardinalities=(4, 2, 3, 5), a term_label=(0, 1, 3), and parent_term_label=(1,).
            The component of the parent label is thus given by (0, 3).
            The shape of the term values is then given by (4, 2, 5).
            The axes of the parent term within the 'value-space' of the term are [1].
            The axes of the complement of the parent term within the 'value-space' of the term are [0, 2].
            The shape of the parent term values is (2,) and its expanded shape in the 'value-space'
            of the term is thus (1, 2, 1).
            The (expanded) shape of the complement of the parent term within the 'value-space' of
            the term is given by (4, 1, 5).
            Consider the parent term values given by np.array([5, 10]), this method would then repeat these
            values along the component axes resulting in the numpy array of shape (4, 2, 5) given by
            array([
                    [
                        [ 5,  5,  5,  5,  5],
                        [10, 10, 10, 10, 10]],
                    [
                        [ 5,  5,  5,  5,  5],
                        [10, 10, 10, 10, 10]],
                    [
                        [ 5,  5,  5,  5,  5],
                        [10, 10, 10, 10, 10]
                    ],
                    [
                        [ 5,  5,  5,  5,  5],
                        [10, 10, 10, 10, 10]
                    ]
                ])


        This method is inspired by:
        https://stackoverflow.com/questions/7656665/how-to-repeat-elements-of-an-array-along-two-axes
        """
        # Determine the shape of the term values (e.g. marginal or factor)
        term_values_shape = self._get_term_values_shape(term_label)
        term_values_axes  = range(len(term_values_shape))

        # Get the axes corresponding to the parent term in the 'value-space' of the term
        # Example: If term_label=(0, 1, 3) and parent_term_label=(1,), then parent_term_axes=[1]
        parent_term_axes = [term_label.index(component) for component in parent_term_label]

        # Determine the expanded shape of the complement of the parent term in the 'value-space' of the term.
        # Example: If term_label=(0, 1, 3), parent_term_label=(1,), and self.cardinalities=(4, 2, 5),
        #          then parent_term_complement_expanded_shape=(4, 1, 5).
        parent_term_complement_expanded_shape = [term_values_shape[axis] if axis not in parent_term_axes else 1 for axis in term_values_axes]
        
        # Construct an identity tensor for the term complement (with the same type as the parent values)
        parent_term_complement_identity_expanded = np.ones(parent_term_complement_expanded_shape, dtype=parent_term_values.dtype)

        # Determine the axes (i.e. components) that are the complements of the parent term and sort them
        # Example: If term_label=(0, 1, 3) and parent_term_label=(1,), the parent term complement is (0, 3)
        #          and thus parent_term_complement_axes=[0, 2]
        parent_term_complement_axes = list(set(term_values_axes)-set(parent_term_axes))
        parent_term_complement_axes.sort()
        
        # Expand the values of the parent term in the 'value-space' of the term
        # Example: If term_label=(0, 1, 3), parent_term_label=(1,), and self.cardinalities=(4, 2, 5),
        #          then parent_term_values_expanded=(1, 2, 1).
        parent_term_values_expanded = np.expand_dims(parent_term_values, axis=parent_term_complement_axes)

        # Use a Kronecker product to repeat the term values along all of the terms complement axes (i.e. components)
        # and return the result.
        # Remark: This step corresponds to the 'value-repetition' along the component.
        return np.kron(parent_term_values_expanded, parent_term_complement_identity_expanded)
    
    def _get_term_values_shape(self, term_label:Tuple[int])->Tuple:
        """
        Return the shape of the values (marginal or factor) of a term that 
        is given by tuple of the cardinalities of each of its components.
        """
        return tuple([self.cardinalities[component] for component in term_label])

    def _get_term_complement_components(self, term_label:Tuple[int])->List:
        """
        Return the complement components of a term. 

        Args:
            term_label (tuple of ints): The term label as tuple of integers.

        Return:
            (list): Sorted list containing the complement components of a term.
                These are the components that do not appear in term.
        
        Example: 
            Consider self.num_components=4 and the term label is (1, 3).
            The components contained in this term label are 1 and 3, while 
            the complement components are 2 and 4. The term complement components 
            of returned by this method would therefore be [2, 4].
    
        """
        # Use set-arithmetics to obtain a list of the components that are
        # not contained in the term, and sort it
        components_set = set(range(self.num_components))
        term_complement_components = list(components_set-set(term_label))
        term_complement_components.sort()

        return term_complement_components

    def _parse_max_term_order(self, max_term_order_label:str, max_term_order:int, upper_limit:int)->int:
        """
        Parse the 'max_term_order'. 
        
        Args:
            max_term_order_label (str): Label for the maximal term order.
            max_term_order (int or None): The maximal term order as integer or as None.
                (Default: None)
            upper_limit (int): Upper limit for the maximal term order.

        Return:
            (int): The parsed maximal term order.
        
        """
        # If the input max_term_order is not specified, use the upper limit
        if max_term_order is None:
            return upper_limit
    
        # Otherwise, check that max_term_order is a positive integer not
        # larger than the upper limit, i.e. in [1, upper_limit]
        if (max_term_order<0) or (upper_limit<max_term_order):
            err_msg = f"The input '{max_term_order_label}' must be either None or [1, {upper_limit}], got '{max_term_order}' instead."
            raise ValueError(err_msg)
        
        return max_term_order

    def _generate_term_labels_dict(self)->Dict[int, Tuple[Tuple]]:
        """
        Generate all possible term labels.

        Args:
            None

        Return:
            (dict of tuples of tuples): Dictionary that has the term
                order (integer) as dictionary-keys and a tuple of the
                terms (that are themselves tuples) of this order as 
                dictionary-values.

        Example:
            Consider self.num_components=3.
            In this case, this method would return the dictionary
            {
                0: ((,),)                    # 1 zero-order term
                1: ((1,), (2,), (3,))        # 3 first-order terms
                2: ((1, 2), (2, 3), (1, 3))  # 3 second-order terms
                3: ((1, 2, 3),)              # 1 third-order term
            }
            where {1, 2, 3} are the one-based components.
        
        Remark: The term labels only depend on the components but not 
                on the size of the cardinalities on these components.
            
        """
        # Construct a list holding the components and determine the powerset of these
        # components that corresponds to an iterable containing all potential terms
        components  = list(range(self.num_components))
        term_labels = self._get_powerset(components)

        # Loop over all terms and construct lists of terms belonging to the same order
        term_labels_dict = collections.defaultdict(list)
        for term_label in term_labels:
            # The order of a term is given by its length.
            # Example: The term (1, 2) has order 2.
            term_order = len(term_label)

            # Only append the term label to the corresponding list (associated with the term orders)
            # in case that the current term order is smaller than the maximal term order
            if term_order<=self.global_max_term_order:
                term_labels_dict[term_order].append(term_label)

        # For all term order, cast the lists containing the term_labels to tuples
        return {term_order: tuple(term_label) for term_order, term_label in term_labels_dict.items()}

    def _get_powerset(self, iterable:Iterable)->Iterable:
        """
        Return the powerset of an iterable as an iterable.
        Example:
        list( powerset([1, 2, 3]) ) => [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
        
        Args:
            iterable (iterable): Iterable of which the powerset should be determined.
        
        Return:
            (iterable): Powerset of the input iterable.
        
        Remarks
        itertools.combinations:
        Returns all (ordered) combinations of elements of a list (involving a fixed number of combinations)
        as an iterable of tuples where each tuple corresponds to one (ordered) combination of elements.
        Example: x=[0, 1, 2] then 
                list( itertools.combinations(x, 0) ) => [()]
                list( itertools.combinations(x, 1) ) => [(0,), (1,), (2,)]
                list( itertools.combinations(x, 2) ) => [(0, 1), (0, 2), (1, 2)]
                list( itertools.combinations(x, 3) ) => [(0, 1, 2)]

        itertools.chain.from_iterable:
        Chains multiple iterables (contained in a an iterable) into one iterable.
        Example: list( itertools.chain.from_iterable([[0, 1, 2], [3, 4], [5, 6, 7, 8]]) ) => [0, 1, 2, 3, 4, 5, 6, 7, 8]
        
        """
        iterable_as_list = list(iterable)
        iterable_length  = len(iterable_as_list)
        
        # Determine the (ordered) combinations with length 0 (combinations_length=0) up to the 
        # combination containing the iterable itself (combinations_length=iterable_length).

        #print('Generating powerset:')
        #return itertools.chain.from_iterable(itertools.combinations(iterable_as_list, combinations_length) for combinations_length in tqdm(range(iterable_length+1)))
        return itertools.chain.from_iterable(itertools.combinations(iterable_as_list, combinations_length) for combinations_length in range(iterable_length+1))

    def _get_numpy_array_index(self, axis_indices:Iterable)->Tuple:
        """
        Return a tuple that can be used to slice numpy arrays
        and that corresponds to the input axis-indices.

        Args:
            axis_indices (iterable): Axis indices for which a
                numpy array should be determined.
            
        Return:
            (tuple): The input axis-indices represented in a
                form that can be used to slice numpy arrays.

        Example:
            If axis_indices=[0, 1, 4], the corresponding numpy-slicing-compatible output 
            of this method would be index would be (array(0), array(1), array(4)).

        """
        return tuple([np.array(axis_index) for axis_index in axis_indices])
