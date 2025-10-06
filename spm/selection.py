import torch
import numpy as np
from scipy.stats import multinomial
import copy

###############################################################################
### DEFINE UTILITY FUNCTIONS HERE
###############################################################################
def multinomial_nll(p_k, n_k, prob_eps=1e-10):
    """
    Calculate the negative log likelihood for a multinomial distribution (in pytorch) for K classes.

    Args:
        p_k (torch.tensor): Tensor of shape (K,) holding the class probabilities.
        n_k (torch.tensor): Tensor of shape (K,) holding the observed counts per class.

    Return:
        (torch.tensor): Negative log likelihood value of shape (1,).
    """
    # Calculate the total number of counts over all classes
    n = torch.sum(n_k)

    # Calculate log(n!) and log(n_k!) using the identity N!=Gamma(N+1) for (0<=N) and the torch function torch.lgamma(N)
    # that calculates log[Gamma(|N+1|)].
    log_n_fac   = torch.lgamma(n+1)
    log_n_k_fac = torch.lgamma(n_k+1)

    # Calculate the negativ logarithm of the normalization term -log(n!*Product_k[1/n_k!]) = Sum_k[log(n_k!)]-log(n!)
    nl_norm_term = torch.sum(log_n_k_fac)-log_n_fac

    # Calculate the cross-entropy term by summing over the classes
    # Remarks: (1) Here p_k are considered probabilities that do not stem from a sigmoid or a softmax function and thus cannot be
    #              represented as logits. Thus torch.nn.functional.cross_entropy cannot be used as it requires logit inputs.
    #          (2) Add a small value to each probability for numerical stability (i.e. to avoid the 'log(0)=-inf' case)
    nl_ce_term = -torch.sum( n_k*torch.log(p_k+prob_eps) )

    # Add these terms to obtain the negative log likelihood and return it
    return nl_norm_term+nl_ce_term

def get_energies(model_outputs, use_softplus=False, device='cuda'):
    """
    Determine the energies associated with each round for a batch of sequences.
    Remark: The lower the energy of a sequence in a certain round the higher
            its probability to be selected in this round, i.e. rho_r(s)=exp(-energy(s))

    Args:
    model_outputs (torch.tensor): Output of the model of shape (#sequences, #rounds),
        assuming that the second axis (i.e. the round axis) corresponds to monotonically
        increasing rounds.
        Assume that the rounds are only the rounds in which selection has been
        performed (i.e. rounds=[1, ..., R] without round 0 if not selection has been
        performed in round 0)
    use_softplus (bool): Should we apply a softplus function to the model outputs
        to map them to [0, +inf] (required if the model ouputs are in [-inf, +inf])
        or not?
        (Default: False)

    Return:
        (torch.tensor): Energies associated with each round that monotonically increase
        with the rounds for the sequences in the batch of shape (#sequences, #rounds+1).
        Add zero energies for round 0 for any sequence (assuming that no selection has
        been performed in round 0).
    """
    # Apply softplus if requested
    if use_softplus:
        # Map the model outputs in [-inf, inf] to positive numbers that are treated as
        # the energy difference between energies of adjacent rounds.
        softplus_fn   = torch.nn.Softplus(beta=1, threshold=20)
        energy_deltas = softplus_fn(model_outputs)
    else:
        energy_deltas = model_outputs

    # Add the energy deltas successively together (i.e. using cummulative sum) along
    # the second (i.e. rounds) axis to obtain the round energies
    energies = torch.cumsum(energy_deltas, dim=1)

    return energies


def get_batch_nll(model_fn, batch_sequences, batch_counts, total_counts, p_init_s, 
                  predict_counts=False, predict_fitness=False, device='cuda', constrained_energy=True, fit_init_round=False):
    """
    Return the negative log-likelihood (nll) for a batch.

    Args:
    model_fn (torch.nn.Module): Model function used as 'model_fn(sequences, round_index)'
        where sequences is as defined below for 'batch_sequences' and round_index is an integer
        specifying the round the model should be evaluated at.
    batch_sequences (torch.tensor): Input representation of the sequences in the batch of shape
        (#batch_sequences, #sequence-features). Can be used as input to model_fn.
    batch_counts (torch.tensor): Number of counts observed for each round and each sequence in
        the batch as torch tensor of shape (#batch_sequences, #rounds).
    total_counts (torch.tensor): Total number of counts observed in any of the rounds as torch
        tensor of shape (#rounds,).
    p_init_s (torch.tensor): Initial sequences distribution over the sequences of the batch as torch
        tensor of shape (#batch_sequences, ).
        Example: For a uniform distribution define
                 p_init_s = torch.ones(batch_sequence.size(0), 1)/batch_sequence.size(0)
    predict_counts (bool): return predicted counts
    constrained_energy (bool): impose monotonically increasing energies over rounds
    fit_init_round (bool): allow model to train on the inital round (Round 0) counts

    Return:
    (torch.tensor): Negative log-likelihood value for the batch sequences and current model as torch tensor
        object of shape (1,).

    """
    # Determine the number of rounds and generate a list of round indices
    num_rounds    = len(total_counts)
    round_indices = list(range(num_rounds))

    if not fit_init_round:
        # Evaluate the model for all batch sequences and for each round (except the first round, i.e. round 0), and
        # concatenate the result along the second (i.e. rounds) axis to a tensor of shape (#batch_sequences, #rounds-1).
        # Remark: We do not evaluate for round 0 because we assume no selection has been performed for this round.
 
            #    x_r = torch.ones(x.shape[0])*r
            #    x_r = x_r[..., None]
        model_output_list = [model_fn(batch_sequences, (torch.ones(batch_sequences.shape[0])*round_index)[...,None].to(device)) for round_index in round_indices[1:]]
    else:
        model_output_list = [model_fn(batch_sequences, (torch.ones(batch_sequences.shape[0])*round_index)[...,None].to(device)) for round_index in round_indices]
    model_outputs = torch.cat(model_output_list, dim=1)

    if constrained_energy: 
        # Determine the energies of the sequences in the batch and rounds [=> torch tensor of shape (#batch_sequences, #rounds)]
        # where the energies are montonously increasing in the rounds (assuming harder selection from round to round) for each
        # sequence and where fitness value of round 0 is set zero for all sequences (i.e. all energy values are negative).
        # Remark: These energies relate to selection factors for each sequence in each round of the form:
        #         selection_factor[s] = exp(-energy[s])
        energies_s_batch = get_energies(model_outputs, use_softplus=False, device=device)
    else:
        energies_s_batch = model_outputs

    if not fit_init_round:
        # Assume that in round 0, all sequences have energy 0
        round_zero_energies = torch.zeros(int(energies_s_batch.size(0)), 1).to(device)
        # Concatenate these round 0 energies and the round >0 energies determined above
        # along the second (i.e. rounds) axis
        energies_s_batch = torch.cat([round_zero_energies, energies_s_batch], dim=1)


    # Renormalize the initial sequence distribution so that it is normalized over the batch using the identity a*b=exp(log(a)+log(b))
    log_p_init_s_batch = (torch.log(p_init_s)-torch.log(torch.sum(p_init_s))).squeeze().to(device)

    # Initialize the (logarithm) of the sequence distribution of the previous round (i.e. before selection in a round)
    # as the initial sequence distribution over the batch
    log_p_r_s_batch_prev = log_p_init_s_batch

    # Loop over the rounds
    nll_batch = torch.tensor([0.0]).to(device)
    predicted_counts = torch.zeros((num_rounds, batch_sequences.shape[0]))
    predicted_fitness = torch.zeros((num_rounds, batch_sequences.shape[0]))
    for round_index in round_indices:
        # Get the number of counts observed for the batch sequences in the current round
        N_r_s_batch = batch_counts[:, round_index].squeeze()

        # Calculate the number of counts observed for the batch sequences in the current round
        # by summing over the batch counts of the current round
        N_r_batch = torch.sum(N_r_s_batch)

        # Get the total number of counts observed over all sequences in the current round
        N_r_tot = total_counts[round_index]

        # Get the energy values of the sequences for the current round and shift all of them so that the
        # minimal energy value in the batch is 0, i.e. all energies are positive.
        # Remark: This step could ensure numerical stability below when updating the within probabilities
        #         but might not be necessary as the energy values should all be positive already.
        energies_r_s_batch  = energies_s_batch[:, round_index]
        fitness_r_s_batch = -energies_r_s_batch  
        #energies_r_s_batch_min = torch.min(energies_r_s_batch)
        energies_r_s_batch_min, _ = torch.min(energies_r_s_batch, dim=0, keepdim=True) # HL: check this
        energies_r_s_batch = energies_r_s_batch - energies_r_s_batch_min
        

        # Update the within-batch sequence distribution (that is normalized within the batch)
        # of the current round using:
        # p_{r}^{batch}(s) = p_{r-1}^{batch}(s)exp[-energy(s)]/(sum_s' p_{r-1}^{batch}(s')exp[-energy(s')] )
        # where 's' denotes a sequence, 'energy(s)' is the energy of a sequence for the
        # current round, and 'p_{r-1}(s)' is p_s_prev here.
        # Remark: We vectorize this expression and use the identity 'a*b=exp(log(a)+log(b))'
        log_p_r_s_batch  = log_p_r_s_batch_prev - energies_r_s_batch
        log_p_r_s_batch = log_p_r_s_batch - torch.log( torch.sum(torch.exp(log_p_r_s_batch)) )

        # Calculate the negative log likelihood (nll) for the current round and
        # add it to the total nll over all rounds
        # Step 1: Multinomial over the observed batch counts (model-parameter-dependent part)
        nll_r_batch  = multinomial_nll(torch.exp(log_p_r_s_batch), N_r_s_batch)

        # Step 2: Multinomial over batch/non-batch counts (model-parameter-independent part)
        nll_r_batch = nll_r_batch + multinomial_nll(torch.tensor([N_r_batch/N_r_tot, 1-N_r_batch/N_r_tot]), torch.tensor([N_r_batch, N_r_tot-N_r_batch]))

        # Step 3: Add it to sum of nll over all rounds
        nll_batch = nll_batch + nll_r_batch

        # log_p_s_batch becomes the log_p_s_batch_pred for the next round
        log_p_r_s_batch_prev = log_p_r_s_batch

       
        if predict_counts:
            predicted_counts[round_index] = torch.exp(log_p_r_s_batch.cpu().detach())*N_r_batch.cpu().detach()
        if predict_fitness:
            predicted_fitness[round_index] = fitness_r_s_batch.cpu().detach()

    if predict_counts and predict_fitness:
        return predicted_counts, predicted_fitness
    elif predict_counts:
        return predicted_counts
    elif predict_fitness:
        return predicted_fitness
    else:
        return nll_batch


def get_batch_fitness(model_fn, batch_sequences, num_rounds, constrained_energy=True, fit_init_round=False, device='cuda'):

    # Determine the number of rounds and generate a list of round indices
    round_indices = list(range(num_rounds))

    if not fit_init_round:
        # Evaluate the model for all batch sequences and for each round (except the first round, i.e. round 0), and
        # concatenate the result along the second (i.e. rounds) axis to a tensor of shape (#batch_sequences, #rounds-1).
        # Remark: We do not evaluate for round 0 because we assume no selection has been performed for this round.
 
            #    x_r = torch.ones(x.shape[0])*r
            #    x_r = x_r[..., None]
        model_output_list = [model_fn(batch_sequences, (torch.ones(batch_sequences.shape[0])*round_index)[...,None].to(device)) for round_index in round_indices[1:]]
    else:
        model_output_list = [model_fn(batch_sequences, (torch.ones(batch_sequences.shape[0])*round_index)[...,None].to(device)) for round_index in round_indices]
    model_outputs = torch.cat(model_output_list, dim=1)

    if constrained_energy: 
        # Determine the energies of the sequences in the batch and rounds [=> torch tensor of shape (#batch_sequences, #rounds)]
        # where the energies are montonously increasing in the rounds (assuming harder selection from round to round) for each
        # sequence and where fitness value of round 0 is set zero for all sequences (i.e. all energy values are negative).
        # Remark: These energies relate to selection factors for each sequence in each round of the form:
        #         selection_factor[s] = exp(-energy[s])
        energies_s_batch = get_energies(model_outputs, use_softplus=False, device=device)
    else:
        energies_s_batch = model_outputs

    if not fit_init_round:
        # Assume that in round 0, all sequences have energy 0
        round_zero_energies = torch.zeros(int(energies_s_batch.size(0)), 1).to(device)
        # Concatenate these round 0 energies and the round >0 energies determined above
        # along the second (i.e. rounds) axis
        energies_s_batch = torch.cat([round_zero_energies, energies_s_batch], dim=1)

    # Loop over the rounds
    '''
    predicted_fitness = torch.zeros((num_rounds, batch_sequences.shape[0]))
    for round_index in round_indices:
        # Get the energy values of the sequences for the current round and shift all of them so that the
        # minimal energy value in the batch is 0, i.e. all energies are positive.
        # Remark: This step could ensure numerical stability below when updating the within probabilities
        #         but might not be necessary as the energy values should all be positive already.
        energies_r_s_batch  = energies_s_batch[:, round_index]
        fitness_r_s_batch = -energies_r_s_batch  

        predicted_fitness[round_index] = fitness_r_s_batch.cpu().detach()
    '''

    predicted_fitness = -energies_s_batch

    return predicted_fitness



def predict_counts(model_fn, batch_sequences, batch_counts, total_counts, p_init_s, 
                  device='cuda', constrained_energy=True, fit_init_round=False, n_samples=1, np_random_seed=0):
    """
    Return the predicted counts of batch sequences

    Args:
    model_fn (torch.nn.Module): Model function used as 'model_fn(sequences, round_index)'
        where sequences is as defined below for 'batch_sequences' and round_index is an integer
        specifying the round the model should be evaluated at.
    batch_sequences (torch.tensor): Input representation of the sequences in the batch of shape
        (#batch_sequences, #sequence-features). Can be used as input to model_fn.
    batch_counts (torch.tensor): Number of counts observed for each round and each sequence in
        the batch as torch tensor of shape (#batch_sequences, #rounds).
    total_counts (torch.tensor): Total number of counts observed in any of the rounds as torch
        tensor of shape (#rounds,).
    p_init_s (torch.tensor): Initial sequences distribution over the sequences of the batch as torch
        tensor of shape (#batch_sequences, ).
        Example: For a uniform distribution define
                 p_init_s = torch.ones(batch_sequence.size(0), 1)/batch_sequence.size(0)
    constrained_energy (bool): impose monotonically increasing energies over rounds
    fit_init_round (bool): allow model to train on the inital round (Round 0) counts
    n_samples: number of samples to draw from the multinomial distribution for all the elements 

    Return:
    (torch.tensor): predicted counts of each sequence in each selection round.

    """

    np.random.seed(np_random_seed)

    # Determine the number of rounds and generate a list of round indices
    num_rounds    = len(total_counts)
    round_indices = list(range(num_rounds))

    if not fit_init_round:
        # Evaluate the model for all batch sequences and for each round (except the first round, i.e. round 0), and
        # concatenate the result along the second (i.e. rounds) axis to a tensor of shape (#batch_sequences, #rounds-1).
        # Remark: We do not evaluate for round 0 because we assume no selection has been performed for this round.
 
            #    x_r = torch.ones(x.shape[0])*r
            #    x_r = x_r[..., None]
        model_output_list = [model_fn(batch_sequences, (torch.ones(batch_sequences.shape[0])*round_index)[...,None].to(device)) for round_index in round_indices[1:]]
    else:
        model_output_list = [model_fn(batch_sequences, (torch.ones(batch_sequences.shape[0])*round_index)[...,None].to(device)) for round_index in round_indices]
    model_outputs = torch.cat(model_output_list, dim=1)

    if constrained_energy: 
        # Determine the energies of the sequences in the batch and rounds [=> torch tensor of shape (#batch_sequences, #rounds)]
        # where the energies are montonously increasing in the rounds (assuming harder selection from round to round) for each
        # sequence and where fitness value of round 0 is set zero for all sequences (i.e. all energy values are negative).
        # Remark: These energies relate to selection factors for each sequence in each round of the form:
        #         selection_factor[s] = exp(-energy[s])
        energies_s_batch = get_energies(model_outputs, use_softplus=False, device=device)
    else:
        energies_s_batch = model_outputs

    if not fit_init_round:
        # Assume that in round 0, all sequences have energy 0
        round_zero_energies = torch.zeros(int(energies_s_batch.size(0)), 1).to(device)
        # Concatenate these round 0 energies and the round >0 energies determined above
        # along the second (i.e. rounds) axis
        energies_s_batch = torch.cat([round_zero_energies, energies_s_batch], dim=1)


    # Renormalize the initial sequence distribution so that it is normalized over the batch using the identity a*b=exp(log(a)+log(b))
    log_p_init_s_batch = (torch.log(p_init_s)-torch.log(torch.sum(p_init_s))).squeeze().to(device)

    # Initialize the (logarithm) of the sequence distribution of the previous round (i.e. before selection in a round)
    # as the initial sequence distribution over the batch
    log_p_r_s_batch_prev = log_p_init_s_batch

    # Loop over the rounds
    predicted_counts = []
    for round_index in round_indices:

        print('Predicting for Round', round_index)

        # Get the number of counts observed for the batch sequences in the current round
        N_r_s_batch = batch_counts[:, round_index].squeeze()

        # Calculate the number of counts observed for the batch sequences in the current round
        # by summing over the batch counts of the current round
        N_r_batch = torch.sum(N_r_s_batch)

        # Get the total number of counts observed over all sequences in the current round
        N_r_tot = total_counts[round_index]

        # Get the energy values of the sequences for the current round and shift all of them so that the
        # minimal energy value in the batch is 0, i.e. all energies are positive.
        # Remark: This step could ensure numerical stability below when updating the within probabilities
        #         but might not be necessary as the energy values should all be positive already.
        energies_r_s_batch  = energies_s_batch[:, round_index]
        fitness_r_s_batch = -energies_r_s_batch  
        #energies_r_s_batch_min = torch.min(energies_r_s_batch)
        energies_r_s_batch_min, _ = torch.min(energies_r_s_batch, dim=0, keepdim=True) # HL: check this
        energies_r_s_batch = energies_r_s_batch - energies_r_s_batch_min
        

        # Update the within-batch sequence distribution (that is normalized within the batch)
        # of the current round using:
        # p_{r}^{batch}(s) = p_{r-1}^{batch}(s)exp[-energy(s)]/(sum_s' p_{r-1}^{batch}(s')exp[-energy(s')] )
        # where 's' denotes a sequence, 'energy(s)' is the energy of a sequence for the
        # current round, and 'p_{r-1}(s)' is p_s_prev here.
        # Remark: We vectorize this expression and use the identity 'a*b=exp(log(a)+log(b))'
        log_p_r_s_batch  = log_p_r_s_batch_prev - energies_r_s_batch
        log_p_r_s_batch = log_p_r_s_batch - torch.log( torch.sum(torch.exp(log_p_r_s_batch)) )

        # Calculate the negative log likelihood (nll) for the current round and
        # add it to the total nll over all rounds
        # Step 1: Multinomial over the observed batch counts (model-parameter-dependent part)
        p_r_s = torch.exp(log_p_r_s_batch)*N_r_batch/N_r_tot
        if N_r_batch < N_r_tot:
            weights = torch.cat([p_r_s, (1-N_r_batch/N_r_tot).reshape(-1)]).cpu().detach()
        else:
            weights = p_r_s.cpu().detach()

        if n_samples<=0:
            predicted_counts.append((torch.exp(log_p_r_s_batch.cpu().detach())*N_r_batch.cpu().detach()).numpy()[None,:])

        else:
            # multinomial.rvs converts weights to numpy.float64
            # and raises error when weights.numpy().sum() > 1
            weights = weights.to(torch.float64).numpy()
            if weights.sum() != 1:
                print(f'Warning: weights.to(torch.float64).numpy().sum() ({weights.sum()}) != 1')
 
                '''
                # correction by subtracting a fixed small value from all elements 
                # not working when the small value is close to min(weights)
                print(f'         subtracting (weights.sum()-1.0) / weights.shape[0] ({(weights.sum()-1.0) / weights.shape[0]}) from each of its element!')
                print(f'         pre-subtraction min(weights)={weights.min()}')
                weights -= (weights.sum()-1.0) / weights.shape[0] 
                '''
 
                # correction by subtracting a small value proportional to the elements
                subtrahend = (weights.sum()-1.0) * weights / weights.sum()
                print(f'         subtracting (weights.sum()-1.0) * weights / weights.sum()')
                print(f'         pre-subtraction min(weights)={weights.min()}, max(weights)={weights.max()}')
                print(f'         min(subtrahend)={subtrahend.min()}, max(subtrahend)={subtrahend.max()}')
                weights -= subtrahend
 
            #np.save(f'weights_{round_index}.npy', weights)
 
            #predicted_counts.append(multinomial.rvs(n=int(N_r_tot.item()), p=weights, size=n_samples).T)
            predicted_counts.append(multinomial.rvs(n=int(N_r_tot.item()), p=weights, size=n_samples))

        #for sample_i in range(n_samples):
        #    print(f'Round{round_index}, sample_i{sample_i}')
        #    N_r_sample = torch.multinomial(weights, int(N_r_tot.item()), replacement=True)
        #    N_r_s_sample = N_r_sample[:-1]
        #    N_r_s_sample_counts = torch.tensor([(N_r_s_sample==x).sum().item() for x in range(p_r_s.shape[0])])
        #    predicted_counts[:,round_index,sample_i] = N_r_s_sample_counts
         
        # log_p_s_batch becomes the log_p_s_batch_pred for the next round
        log_p_r_s_batch_prev = log_p_r_s_batch

    predicted_counts = np.array(predicted_counts)
    predicted_counts = np.moveaxis(predicted_counts, 0, 2)

    return predicted_counts
