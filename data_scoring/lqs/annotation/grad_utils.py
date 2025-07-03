from torch.func import grad, jvp, vmap
from .model_wrapper import TransformerWrapper
import torch


def jvp_single(input_ids, attention_mask, labels, loss_mask, model: TransformerWrapper, lam_param, params, buffers):
    loss_single_func_wrapper = lambda params: \
        model.compute_loss_func_single(
            params, buffers, model,
            input_ids, attention_mask, labels, loss_mask)
    _, ct = jvp(loss_single_func_wrapper, (params,), (lam_param,))
    #print(_.shape, ct.shape)

    return ct

def jvp_single_verify(input_ids, attention_mask, labels, loss_mask, 
               model: TransformerWrapper, lam_param, params, buffers):
    import time
    loss_single_func_wrapper = lambda params: \
        model.compute_loss_func_single(
            params, buffers, model,
            input_ids, attention_mask, labels, loss_mask)
    
    start_time = time.time()
    _, ct = jvp(loss_single_func_wrapper, (params,), (lam_param,))
    if torch.cuda.is_available():
        torch.cuda.synchronize() 
    jvp_time = time.time() - start_time
    
    start_time = time.time()
    grad_params = grad(loss_single_func_wrapper)(params)
    grad_flat = torch.cat([g.view(-1) for g in grad_params.values()])
    lam_flat = torch.cat([v.view(-1) for v in lam_param.values()])
    ct_manual = torch.dot(grad_flat, lam_flat)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    grad_time = time.time() - start_time

    print(f"\n[jvp] Time: {jvp_time:.6f}s | [grad] Time: {grad_time:.6f}s")
    print(f"Difference (ct - ct_manual): " + str(ct - ct_manual))

    return ct



def jvp_single_m(input_ids, attention_mask, labels, loss_mask, model: TransformerWrapper, lam_param, params, buffers, theta1, m):
    loss_single_func_wrapper = lambda params: \
        model.compute_loss_func_single(
            params, buffers, model,
            input_ids, attention_mask, labels, loss_mask)
    # _, ct = jvp(loss_single_func_wrapper, (params,), (lam_param,))

    grad_params = grad(loss_single_func_wrapper)(params)
    grad_flat = torch.cat([g.view(-1) for g in grad_params.values()])
    lam_flat = torch.cat([v.view(-1) for v in lam_param.values()])
    ct_manual = torch.dot(grad_flat, lam_flat)

    # hyper
    theta2 = torch.pi / 2 - theta1

    # cos
    grad_norm = torch.norm(grad_flat)
    lam_norm = torch.norm(lam_flat)
    cos_theta = ct_manual / (grad_norm * lam_norm +  1e-8) 
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

    theta = torch.acos(cos_theta)

    # threshold
    mask_loose = theta < theta1          
    mask_normal = (theta >= theta1) & (theta <= theta2)  
    mask_strict = theta > theta2        

    m1 = torch.zeros_like(theta)
    m2 = torch.zeros_like(theta)

    # small angle
    m_loose = torch.tensor(-m, device=theta.device)  
    m1_loose = torch.clamp(0.1*torch.tanh(m_loose), 0, 1)  
    m2_loose = -torch.abs(m_loose) 
    m1 = torch.where(mask_loose, m1_loose, m1) 
    m2 = torch.where(mask_loose, m2_loose, m2) 

    # middle angle
    m1_normal = 1
    m2_normal = 0
    m1 = torch.where(mask_normal, m1_normal, m1)  
    m2 = torch.where(mask_normal, m2_normal, m2) 

    # big angle
    m_strict = torch.tensor(m, device=theta.device)  
    m1_strict = torch.clamp(1 + 0.1*torch.tanh(m_strict), 1, 2) 
    m2_strict = torch.abs(m_strict)
    m1 = torch.where(mask_strict, m1_strict, m1) 
    m2 = torch.where(mask_strict, m2_strict, m2)

    angle = torch.clamp(m1 * theta + m2, 0, torch.pi)
    total_ct_cos = grad_norm * lam_norm * torch.cos(angle)
    
    return total_ct_cos



def jvp_single_w(input_ids, attention_mask, labels, loss_mask, model: TransformerWrapper, lam_param, params, buffers, w):
    loss_single_func_wrapper = lambda params: \
        model.compute_loss_func_single(
            params, buffers, model,
            input_ids, attention_mask, labels, loss_mask)
    _, ct = jvp(loss_single_func_wrapper, (params,), (lam_param,))

    grad_params = grad(loss_single_func_wrapper)(params)
    grad_flat = torch.cat([g.view(-1) for g in grad_params.values()])
    grad_norm = torch.norm(grad_flat) + 1e-8 
    
    total_ct = ct * (1 + w / grad_norm)
    
    return total_ct

def jvp_single_grad(input_ids, attention_mask, labels, loss_mask, model: TransformerWrapper, lam_param, params, buffers, grad_checkpoint):
    loss_single_func_wrapper = lambda params: \
        model.compute_loss_func_single(
            params, buffers, model,
            input_ids, attention_mask, labels, loss_mask)

    grad_params = grad(loss_single_func_wrapper)(params)
    grad_flat = torch.cat([g.view(-1) for g in grad_params.values()])
    lam_flat = torch.cat([v.view(-1) for v in lam_param.values()])

    ct_manual = torch.dot(grad_flat, lam_flat)

    grad_norm = torch.norm(grad_flat)
    lam_norm = torch.norm(lam_flat)
    cos_theta = ct_manual / (grad_norm * lam_norm)
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)  
    # theta = torch.acos(cos_theta) 

    # total_ct_cos = lam_norm * (grad_checkpoint / grad_norm) * cos_theta
    total_ct_cos = lam_norm * (grad_norm / grad_checkpoint) * cos_theta

    return total_ct_cos, grad_norm

def jvp_single_vis(input_ids, attention_mask, labels, loss_mask, model: TransformerWrapper, lam_param, params, buffers):
    loss_single_func_wrapper = lambda params: \
        model.compute_loss_func_single(
            params, buffers, model,
            input_ids, attention_mask, labels, loss_mask)

    grad_params = grad(loss_single_func_wrapper)(params)
    grad_flat = torch.cat([g.view(-1) for g in grad_params.values()])
    lam_flat = torch.cat([v.view(-1) for v in lam_param.values()])

    ct_manual = torch.dot(grad_flat, lam_flat)

    grad_norm = torch.norm(grad_flat)
    lam_norm = torch.norm(lam_flat)
    cos_theta = ct_manual / (grad_norm * lam_norm)
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)  
    theta = torch.acos(cos_theta) 

    # total_ct_cos = lam_norm * (grad_checkpoint / grad_norm) * cos_theta

    return ct_manual, lam_norm, grad_norm, cos_theta, theta


def jvp_verify(input_ids, attention_mask, labels, loss_mask, model: TransformerWrapper, lam_param, params, buffers, w):
    loss_single_func_wrapper = lambda params: \
        model.compute_loss_func_single(
            params, buffers, model,
            input_ids, attention_mask, labels, loss_mask)
    
    _, ct_jvp = jvp(loss_single_func_wrapper, (params,), (lam_param,))

    grad_params = grad(loss_single_func_wrapper)(params)
    grad_flat = torch.cat([g.view(-1) for g in grad_params.values()])
    lam_flat = torch.cat([v.view(-1) for v in lam_param.values()])

    ct_manual = torch.dot(grad_flat, lam_flat)

    # cos
    grad_norm = torch.norm(grad_flat)
    lam_norm = torch.norm(lam_flat)
    cos_theta = ct_manual / (grad_norm * lam_norm)
    ct_cos = grad_norm * lam_norm * cos_theta
    

    theta = torch.acos(cos_theta)
    theta_degrees = theta * (180 / torch.pi)

    print('\nct_manual: ', ct_manual)
    print('\nct_cos: ', ct_cos)
    # assert torch.allclose(ct_jvp, ct_manual)

    return ct_manual

def jvp_batch(model: TransformerWrapper, batch, lam_param, params, buffers, chunk_size=None, w=None, theta1=None, m=None, grad_checkpoint=None):
    # if w != None:
    #     return vmap(jvp_single_w, in_dims=(0, 0, 0, 0, None, None, None, None, None), chunk_size=chunk_size)(
    #         batch["input_ids"], batch["attention_mask"], batch["label"], batch["loss_mask"], model, lam_param, params, buffers, w)
    # elif theta1 != None:
    #     return vmap(jvp_single_m, in_dims=(0, 0, 0, 0, None, None, None, None, None, None), chunk_size=chunk_size)(
    #         batch["input_ids"], batch["attention_mask"], batch["label"], batch["loss_mask"], model, lam_param, params, buffers, theta1, m)
    # elif grad_checkpoint != None:
    #     return vmap(jvp_single_grad, in_dims=(0, 0, 0, 0, None, None, None, None, None), chunk_size=chunk_size)(
    #         batch["input_ids"], batch["attention_mask"], batch["label"], batch["loss_mask"], model, lam_param, params, buffers, grad_checkpoint)
    # else:
    # return vmap(jvp_single_vis, in_dims=(0, 0, 0, 0, None, None, None, None), chunk_size=chunk_size)(
        # batch["input_ids"], batch["attention_mask"], batch["label"], batch["loss_mask"], model, lam_param, params, buffers)

    return vmap(jvp_single_grad, in_dims=(0, 0, 0, 0, None, None, None, None, None), chunk_size=chunk_size)(
        batch["input_ids"], batch["attention_mask"], batch["label"], batch["loss_mask"], model, lam_param, params, buffers, grad_checkpoint)


def hvp_fwdrev(model: TransformerWrapper, batch, lam_param, params, buffers, bs, gacc, ws):
    f = model.compute_loss_func
    def grad_wrapper(pr):
        g = {n: 0 for n in params}
        for i in range(gacc):
            mini_batch = {k: v[i*bs:(i+1)*bs] for k, v in batch.items()}
            _g = grad(f)(pr, buffers, model, **mini_batch)
            for n in g:
                g[n] += _g[n]
        return g
    _, hvp_res = jvp(grad_wrapper, (params,), (lam_param,))
    hvp_res = model.params_to_vector(hvp_res) / (ws * gacc)
    return hvp_res
