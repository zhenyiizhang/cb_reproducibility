import torch, math


def cal_interaction(z, lnw, interaction_potential, t, m=16, threshold=1000, use_mass = True, mass_detach = False):
    """
    Calculate interaction forces between particles, divided into groups of m particles with neighbor normalization.
    When the distance between two particles is greater than threshold, their interaction force is set to 0.

    Parameters:
        z: Tensor of shape (batch_size, embed_dim), representing particle positions or embedding vectors.
        lnw: Tensor of shape (batch_size, ), representing particle weights in log form.
        interaction_potential: Potential function that takes input of shape [num_pairs, embed_dim] and outputs gradients.
        m: Number of particles per group, must be m>=2.
        threshold: Scalar, if Euclidean distance between two particles exceeds this value, force is set to 0.

    Process:
        1. Calculate actual mass w from lnw and normalize to prevent force scale from varying with particle count.
        2. Randomly shuffle particle order, then divide into groups of m particles, error if remaining particles = 1.
        3. For each group:
           - Use torch.triu_indices to get indices for all particle pairs (computed once);
           - Calculate diff between each pair (shape [num_pairs, embed_dim]), call interaction_potential
             and compute gradients via autodiff (negative gradient as force f);
           - Calculate Euclidean distance from diff, create mask where 1 indicates distance <= threshold,
             for each (i,j) pair, add f * mass[j] to particle i's force, subtract f * mass[i] from particle j's force;
           - Normalize each particle by neighbor count (m-1).
        4. Return net_force after restoring forces to original particle order.
    """
    if m < 2:
        raise ValueError("Number of particles per group m must be greater than or equal to 2.")
    
    # Calculate actual mass and normalize (multiply by total particle count)
    lnw = lnw.detach()
    w = torch.exp(lnw)
    # w = w/w.sum()
    n = w.shape[0]
    w = w * n
    # w = torch.ones_like(w)

    batch_size, embed_dim = z.shape
    device = z.device

    if batch_size < 2:
        raise ValueError("Total number of particles must be at least 2.")

    # Randomly shuffle particle order
    perm = torch.randperm(batch_size, device=device)
    z_shuffled = z[perm]

    if batch_size % m == 0:
        # Divide into complete groups and remainder
        num_full_groups = batch_size // m    # Number of complete groups, each with m particles
        remainder = batch_size % m           # Number of remaining particles
    else:
        if batch_size < m:
            num_full_groups = 0
            remainder = batch_size  
        else:
            num_full_groups = batch_size // m - 1    # Number of complete groups, each with m particles
            remainder = batch_size % m + m          # Number of remaining particles

    # Error if remainder group has only 1 particle
    if remainder == 1:
        raise ValueError("Remainder group contains only 1 particle, isolated particles not allowed!")

    # Initialize total force tensor
    net_force = torch.zeros_like(z)

    for i in range(num_full_groups):
        groups_z = z_shuffled[i * m:(i + 1) * m]
        groups_w = w[perm[i * m:(i + 1) * m]]
        groups_lnw = torch.log(groups_w / groups_z.shape[0])

        force_groups = interaction_potential(groups_z, groups_lnw, t)
        net_force[perm[i * m:(i + 1) * m]] = force_groups
    
    if remainder > 0:
        groups_z = z_shuffled[num_full_groups * m:]
        groups_w = w[perm[num_full_groups * m:]]
        groups_lnw = torch.log(groups_w / groups_z.shape[0])

        force_groups = interaction_potential(groups_z, groups_lnw, t)
        net_force[perm[num_full_groups * m:]] = force_groups
    
    # import pdb; pdb.set_trace()

    # if num_full_groups > 0:
    #     groups_z = z_shuffled[:num_full_groups * m].view(num_full_groups, m, embed_dim)
    #     idx = torch.triu_indices(m, m, offset=1, device=device)
    #     num_pairs = idx.shape[1]
    #     diffs = groups_z[:, idx[0], :] - groups_z[:, idx[1], :]
    #     mask = (diffs.norm(dim=-1) <= threshold).float().unsqueeze(-1)
    #     diffs_flat = diffs.reshape(num_full_groups * num_pairs, embed_dim)
    #     diffs_flat.requires_grad_(True)
    #     potentials = interaction_potential(diffs_flat)
    #     grad_potentials = torch.autograd.grad(
    #         outputs=potentials,
    #         inputs=diffs_flat,
    #         grad_outputs=torch.ones_like(potentials),
    #         create_graph=True,
    #     )[0]
    #     f_pairs_flat = -grad_potentials
    #     f_pairs = f_pairs_flat.view(num_full_groups, num_pairs, embed_dim)
    #     f_pairs = f_pairs * mask
    #     mass_groups = w[perm[:num_full_groups * m]].view(num_full_groups, m, 1)
    #     force_groups = torch.zeros(num_full_groups, m, embed_dim, device=device)
    #     i_idx = idx[0].unsqueeze(0).expand(num_full_groups, -1)
    #     j_idx = idx[1].unsqueeze(0).expand(num_full_groups, -1)
    #     mass_j = mass_groups.gather(1, j_idx.unsqueeze(-1))
    #     if use_mass:
    #         contrib_i = f_pairs * mass_j
    #     else:
    #         contrib_i = f_pairs
    #     force_groups.scatter_add_(1, i_idx.unsqueeze(-1).expand(-1, -1, embed_dim), contrib_i)
    #     mass_i = mass_groups.gather(1, i_idx.unsqueeze(-1))
    #     if use_mass:
    #         contrib_j = -f_pairs * mass_i
    #     else:
    #         contrib_j = -f_pairs
    #     force_groups.scatter_add_(1, j_idx.unsqueeze(-1).expand(-1, -1, embed_dim), contrib_j)
    #     force_groups = force_groups / (m - 1)
    #     net_force[perm[:num_full_groups * m]] = force_groups.reshape(num_full_groups * m, embed_dim)
    # if remainder > 0:
    #     group_z = z_shuffled[num_full_groups * m:]
    #     idx_rem = torch.triu_indices(remainder, remainder, offset=1, device=device)
    #     diffs = group_z[idx_rem[0]] - group_z[idx_rem[1]]
    #     mask_rem = (diffs.norm(dim=-1) <= threshold).float().unsqueeze(-1)
    #     diffs.requires_grad_(True)
    #     potentials = interaction_potential(diffs)
    #     grad_potentials = torch.autograd.grad(
    #         outputs=potentials,
    #         inputs=diffs,
    #         grad_outputs=torch.ones_like(potentials),
    #         create_graph=True,
    #     )[0]
    #     f_pairs = -grad_potentials
    #     f_pairs = f_pairs * mask_rem
    #     mass_group = w[perm[num_full_groups * m:]].view(remainder, 1)
    #     force_group = torch.zeros(remainder, embed_dim, device=device)
    #     i_idx_rem = idx_rem[0]
    #     j_idx_rem = idx_rem[1]
    #     mass_j = mass_group[j_idx_rem]
    #     if use_mass:
    #         contrib_i = f_pairs * mass_j
    #     else:
    #         contrib_i = f_pairs
    #     force_group.scatter_add_(0, i_idx_rem.unsqueeze(-1).expand(-1, embed_dim), contrib_i)
    #     mass_i = mass_group[i_idx_rem]
    #     if use_mass:
    #         contrib_j = -f_pairs * mass_i
    #     else:
    #         contrib_j = -f_pairs
    #     force_group.scatter_add_(0, j_idx_rem.unsqueeze(-1).expand(-1, embed_dim), contrib_j)
    #     force_group = force_group / (remainder - 1)
    #     net_force[perm[num_full_groups * m:]] = force_group
    
    return net_force

    # return net_force


def euler_sdeint(sde, initial_state, dt, ts):
    device = initial_state[0].device
    # Initial time, based on first timepoint in ts
    t0 = ts[0].item()
    tf = ts[-1].item()
    current_state = initial_state
    current_time = t0

    output_states = []  # Store states at each ts timepoint
    ts_list = ts.tolist()
    next_output_idx = 0
    # Continue integration while current time hasn't exceeded tf
    while current_time <= tf + 1e-8:
        # If current time reaches or exceeds next output time, record current state
        if current_time >= ts_list[next_output_idx] - 1e-8:
            output_states.append(current_state)
            next_output_idx += 1
            # Exit if all output times have been recorded
            if next_output_idx >= len(ts_list):
                break
        t_tensor = torch.tensor([current_time], device=device)
        # Calculate drift part (f_z, f_lnw) = f(t, y)
        f_z, f_lnw = sde.f(t_tensor, current_state)
        # Calculate diffusion term: generate random noise for z and lnw (noise variance = dt)
        noise_z = torch.randn_like(current_state[0]) * math.sqrt(dt)
        g_z = sde.g(t_tensor, current_state[0])
        # Euler–Maruyama update
        new_z = current_state[0] + f_z * dt + g_z * noise_z
        new_lnw = current_state[1] + f_lnw * dt 

        current_state = (new_z, new_lnw)
        
        current_time += dt

    # Fill remaining output times if any
    while len(output_states) < len(ts_list):
        output_states.append(current_state)
    
    # Organize list into tensors, note states are tuples (z, lnw)
    traj_z = torch.stack([state[0] for state in output_states], dim=0)
    traj_lnw = torch.stack([state[1] for state in output_states], dim=0)
    return traj_z, traj_lnw

def euler_sdeint_split(sde, initial_state, dt, ts, noise_std = 0.01):
    device = initial_state[0].device
    # Initial time, based on first timepoint in ts
    t0 = ts[0].item()
    tf = ts[-1].item()
    current_state = initial_state
    current_time = t0

    output_states = []  # Store states at each ts timepoint
    ts_list = ts.tolist()
    next_output_idx = 0
    w_prev = torch.exp(current_state[1])  # Weight from previous timestep
    # Continue integration while current time hasn't exceeded tf
    while current_time <= tf + 1e-8:
        
        t_tensor = torch.tensor([current_time], device=device)
        # Calculate drift part (f_z, f_lnw) = f(t, y)
        f_z, f_lnw = sde.f(t_tensor, current_state)
        # Calculate diffusion term: generate random noise for z and lnw (noise variance = dt)
        noise_z = torch.randn_like(current_state[0]) * math.sqrt(dt)
        g_z = sde.g(t_tensor, current_state[0])
        # Euler–Maruyama update
        new_z = current_state[0] + f_z * dt + g_z * noise_z
        new_lnw = current_state[1] + f_lnw * dt

        current_time += dt

        # If current time reaches or exceeds next output time, record current state
        if current_time >= ts_list[next_output_idx] - 1e-8:
            w_next = torch.exp(new_lnw)  # Weight at current timestep
            r = w_next / w_prev  # Weight change ratio
            
            # 改进思路：
            # 1. 先将r分为两类：r >= 1（需要split）和 r < 1（可能灭绝）。
            # 2. 对于r >= 1的粒子，计算每个粒子需要生成多少个后代（m_j），
            #    这里可以先对所有r >= 1的粒子一次性采样，得到每个粒子的m_j，然后repeat和加噪声。
            # 3. 对于r < 1的粒子，直接一次性采样一个0-1的随机数，判断是否保留。
            # 4. 最后将所有新粒子的z和lnw拼接起来即可。

            # 1. 分类
            mask_split = (r >= 1).squeeze()  # 需要split的mask
            mask_extinct = ~mask_split       # 需要判断灭绝的mask

            # 2. split部分
            if mask_split.any():
                r_split = r[mask_split]
                new_z_split = new_z[mask_split]
                new_lnw_split = new_lnw[mask_split]
                # 计算每个粒子的floor和小数部分
                r_floor = torch.floor(r_split)
                r_frac = r_split - r_floor
                # 对每个粒子采样是否+1
                rand_frac = torch.rand_like(r_frac)
                m_j = r_floor.int().squeeze() + (rand_frac < r_frac).int().squeeze()
                # 只保留m_j>0的粒子
                valid_mask = m_j > 0
                m_j = m_j[valid_mask]
                if m_j.numel() > 0:
                    # repeat粒子
                    repeated_z = torch.repeat_interleave(new_z_split[valid_mask], m_j, dim=0)
                    repeated_lnw = torch.repeat_interleave(new_lnw_split[valid_mask], m_j, dim=0)
                    # 加噪声
                    noise = torch.normal(0, noise_std, size=repeated_z.shape, device=device)
                    split_z = repeated_z + noise
                    split_lnw = repeated_lnw
                else:
                    split_z = torch.empty(0, new_z.shape[1], device=device)
                    split_lnw = torch.empty(0, 1, device=device)
            else:
                split_z = torch.empty(0, new_z.shape[1], device=device)
                split_lnw = torch.empty(0, 1, device=device)

            # 3. extinction部分
            if mask_extinct.any():
                r_extinct = r[mask_extinct]
                new_z_extinct = new_z[mask_extinct]
                new_lnw_extinct = new_lnw[mask_extinct]
                rand_keep = torch.rand_like(r_extinct)
                keep_mask = (rand_keep < r_extinct)
                # 检查keep_mask形状，确保是一维
                if keep_mask.dim() > 1 and keep_mask.shape[-1] == 1:
                    keep_mask = keep_mask.squeeze(-1)
                if keep_mask.any():
                    extinct_z = new_z_extinct[keep_mask]
                    extinct_lnw = new_lnw_extinct[keep_mask]
                else:
                    extinct_z = torch.empty(0, new_z.shape[1], device=device)
                    extinct_lnw = torch.empty(0, 1, device=device)
            else:
                extinct_z = torch.empty(0, new_z.shape[1], device=device)
                extinct_lnw = torch.empty(0, 1, device=device)

            # 4. 合并
            if split_z.shape[0] > 0 or extinct_z.shape[0] > 0:
                new_z = torch.cat([split_z, extinct_z], dim=0)
                new_lnw = torch.cat([split_lnw, extinct_lnw], dim=0)
                new_lnw = torch.log(torch.ones(new_z.shape[0], 1, device=device) / initial_state[0].shape[0])
            else:
                new_z = torch.empty(0, current_state[0].shape[1], device=device)
                new_lnw = torch.empty(0, 1, device=device)
            current_state = (new_z, new_lnw)
            output_states.append(current_state)
            next_output_idx += 1
            w_prev = torch.exp(new_lnw)
            # Exit if all output times have been recorded
            if next_output_idx >= len(ts_list):
                break
        else:
            current_state = (new_z, new_lnw)

    # Fill remaining output times if any
    while len(output_states) < len(ts_list):
        output_states.append(current_state)
    
    # Organize list into tensors, note states are tuples (z, lnw)
    traj_z = [state[0] for state in output_states]
    traj_lnw = [state[1] for state in output_states]
    return traj_z, traj_lnw