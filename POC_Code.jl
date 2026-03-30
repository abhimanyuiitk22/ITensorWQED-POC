using ITensors
using QuantumOptics
using WaveguideQED
using ITensorMPS

times = 0:0.01:10
dt = times[2] - times[1]
bw = WaveguideBasis(1,times)
#Basis of two-level-system
be = FockBasis(1)
#Define waveguide operators
w = destroy(bw)
wd = create(bw)
#Rasing and lowering operators
s = destroy(be)
sd = create(be)
#Hamiltonian
γ = 1.0
H = im*sqrt(γ/dt)*(sd ⊗ w - s ⊗ wd)



#TLS+onephoton to MPS 
function OnePhotonAndTLS(tls_site, wg_sites, view)
    N_wg = length(wg_sites)
    
    all_sites = [tls_site; wg_sites]
    N = length(all_sites)
    ψ_mps = MPS(all_sites)
    
    links = [Index(2, "Link,l=$i") for i in 1:N-1]
    
    
    T1 = ITensor(all_sites[1], links[1])
    T1[all_sites[1]=>1, links[1]=>1] = 1.0 
    ψ_mps[1] = T1
    
    
    for n in 2:N
        s = all_sites[n]
        ξ = view[n-1] 
        
        li = links[n-1]
        
        if n < N
            lo = links[n]
            T = ITensor(li, s, lo)
            
            T[li=>1, s=>1, lo=>1] = 1.0  
            T[li=>1, s=>2, lo=>2] = ξ    
            T[li=>2, s=>1, lo=>2] = 1.0  
            ψ_mps[n] = T
        else
            # Last site
            T = ITensor(li, s)
            T[li=>1, s=>2] = ξ    
            T[li=>2, s=>1] = 1.0 
            ψ_mps[n] = T
        end
    end
    orthogonalize!(ψ_mps, 1)
    return ψ_mps
end

tls_site = siteind("S=1/2")
wg_sites = siteinds("S=1/2", length(times))



ξ(t,τG,t0) = sqrt(2/τG)*(log(2)/pi)^(1/4)*
exp(-2*log(2)*(t-t0)^2/τG^2)
τG,t0 = 1,5

ψ_w = onephoton(bw,ξ,τG,t0)

ψ = OnePhotonAndTLS(tls_site, wg_sites, OnePhotonView(ψ_w))
#Time Evolution Loop
for k in 1:(length(wg_sites))
    
    s_tls = siteind(ψ, k)
    s_bin = siteind(ψ, k+1)
    T_s,T_sd   = ITensors.op("S+", s_tls), ITensors.op("S-", s_tls)
    T_w,T_wd   = ITensors.op("S+", s_bin), ITensors.op("S-", s_bin)  
    H_local =  sqrt(γ/dt) * (T_sd * T_w - T_s * T_wd)
    U_gate  = exp(H_local * dt)

  
    two_site_tensor = ψ[k] * ψ[k+1]
    
   
    acted_tensor = noprime(U_gate * two_site_tensor)
    
    
    left_inds = (k == 1) ? [s_bin] : [linkind(ψ, k-1), s_bin]
    
    U, S, V = svd(acted_tensor, left_inds; lefttags="Link,l=$k")
    
    ψ[k] = U
    ψ[k+1] = S * V
    
   
end

photon_probs = ITensorMPS.expect(ψ, "ProjDn"; site_range=1:(length(ψ)-1))/(dt)
tls_final_val = ITensorMPS.expect(ψ, "ProjDn"; site_range=length(ψ):length(ψ))[1]
time_axis = times[1:length(photon_probs)]
input_view = OnePhotonView(ψ_w)
input_probs = [abs2(input_view[n]/(dt)^0.5) for n in 1:length(input_view)]

p1 = plot(time_axis, input_probs,
    title="Input vs Scattered Pulse from MPS Simulation",
    xlabel="Time (t)", 
    ylabel="Probability |ξ(t)|²",
    label="Input Pulse",
    lw=2,
    color=:red,
    ls=:dash)

plot!(p1, time_axis, photon_probs,
    label="Scattered Pulse",
    lw=1,
    color=:blue,
    fill=(0, 0.2, :blue))

savefig(p1, joinpath(@__DIR__, "input_vs_scattered_pulse.png"))
display(p1)
