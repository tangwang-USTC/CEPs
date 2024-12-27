
using Plots,DataFrames,CSV,Format,OhMyREPL

pathroot = "C:/Users/Administrator/.julia/pkgGithub/CPEs"
pathdatas = "D:/atom/datas/2024/CEPs"

datatype = BigFloat
datatype = Float64

include(joinpath(pathroot,"Mathematics/maths.jl"))
include(joinpath(pathroot,"src/ES/ESs.jl"))
include(joinpath(pathroot,"test/run_collisions/algorithm/modules.jl"))
include(joinpath(pathroot,"test/run_collisions/paras_alg_optim.jl"))

datatype = BigFloat
# datatype = Float64
is_change_datatype = true
is_change_datatype = false

rtol_OrjL = 1e-14
RDnuT = -1e-5

is_re_seed = true
is_re_seed = false
is_show_nuTi = false
is_optim =  true
        
is_C = true 
# is_C = false 

is_norm_uhL = true
is_norm_uhL = false
        
    is_Jacobian = true
    # is_Jacobian = false
    is_Hessian = false
    is_constraint = false
    is_bs = false

nModL = 3         # when `L` is odd number, the optimization is more challenge!
if nModL == 3
    naiLok = datatype.([0.4429805367270422470174924231902531025012880842972989134553914903140590230558192
            0.2266138115283708717468598063515828621194924776557254115578782704094342418434013
            0.3304056517445868812356477704581640353792194380469756749867302392765067351007709])
    uaiLok = datatype.([0.05034178675563209889943384505709961613170249352291586568335469388449438035259866
            -0.00330104437197134090547557153815187183884883604346171135816840427024951454698761
            0.2000000000000000000000000000000000000000000000000000000000000000000000000000004])
    vthiLok = datatype.([2.172107436802767216842088574135748210446180805776723195502077652793266120892492
            0.3964075780707377171697420092387201548468603708624627768275445691579439979825734
            0.4314849851264950659881694166255316347069588233608140276703777780487898811249299])

    # naiL = deepcopy(naiLok)
    # uaiL = deepcopy(uaiLok)
    # vthiL = deepcopy(vthiLok)
end
njL = 3 * nModL

# is_anasys_L = false
is_anasys_L = true
if is_anasys_L
    nL =  (29 - 1) / 2 + 1 |> Int
    errMhjL = zeros(njL,nL)
    errnuTM = zeros(nL,4)
    

    is_show_Dc = false
    if 1 == 1

        L = 1 + 0
        include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_Anasis_kerl.jl"))
        
        is_re_seed = false
        L = 1 + 2
        include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_Anasis_kerl.jl"))
        L = 1 + 4
        include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_Anasis_kerl.jl"))
        L = 1 + 6
        include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_Anasis_kerl.jl"))
        L = 1 + 8
        include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_Anasis_kerl.jl"))
        L = 1 + 10
        is_show_nuTi = true
        include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_Anasis_kerl.jl"))
        L = 1 + 12
        include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_Anasis_kerl.jl"))
        L = 1 + 14
        include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_Anasis_kerl.jl"))
        L = 1 + 16
        include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_Anasis_kerl.jl"))
        L = 1 + 18
        include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_Anasis_kerl.jl"))
        L = 1 + 20
        include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_Anasis_kerl.jl"))
        L = 1 + 22
        include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_Anasis_kerl.jl"))
        L = 1 + 24
        include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_Anasis_kerl.jl"))
        L = 1 + 26
        include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_Anasis_kerl.jl"))
        L = 1 + 28
        is_show_nuTi = true
        include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_Anasis_kerl.jl"))
        
    end
    if is_anasys_L

        filefigs = string("is_C",Int(is_C), "_is_uN", Int(is_norm_uhL), "_is_J", Int(is_Jacobian),"_RDnuT=",RDnuT)
        figname = string("nMod",nModL, "_uhN", fmtf2(norm(uaiL)), "_ThN", fmtf2(norm(vthiL)))
        file_fig_fold = string(joinpath(pathdatas,"Analysis/optim_convergence",figname))
        ispath(file_fig_fold) || mkpath(file_fig_fold)
        filename = string(joinpath(file_fig_fold,filefigs))

        titleMhjL = string("is_C,is_uhLN,is_Jacob=", Int.([is_C,is_norm_uhL,is_Jacobian]), ", RDnuT=",RDnuT)
        Lvec = 1:2:29
        label = string.("L=",reshape(Lvec,1,15))
        xxx = (jvecL .- L) |> Vector{Int}
        perrMhjL = plot(xxx,errMhjL,line=(2,:auto),label=label)
        xlabel!("j - L")
        ylabel!("err_MhjL")
        title!(titleMhjL)

        label = string.("j-L=",reshape(jvecL .- L,1,njL))
        perrMhjL_L = plot(Lvec,reshape(errMhjL,nL,njL),line=(2,:auto),label=label)
        # xlabel!("L")
        ylabel!("err_MhjL")
    
        label = ["Dn" "DT" "DEk" "DMh"]
        perrnuTM = plot(Lvec,errnuTM,line=(2,:auto),label=label)
        xlabel!("L")
        ylabel!("err_nuTM")
        display(plot(perrMhjL,perrMhjL_L,perrnuTM,layout=(3,1)))

        plot(perrMhjL,perrMhjL_L,perrnuTM,layout=(3,1))

        savefig(string(filename,"_Leven.png"))

    end
else
    if 1 == 1

        L = 1 + 0
        include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_kerl.jl"))
        
        is_re_seed = false
        L = 1 + 2
        include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_kerl.jl"))
        L = 1 + 4
        include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_kerl.jl"))
        L = 1 + 6
        include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_kerl.jl"))
        L = 1 + 8
        include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_kerl.jl"))
        L = 1 + 10
        is_show_nuTi = true
        include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_kerl.jl"))
        L = 1 + 12
        include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_kerl.jl"))
        L = 1 + 14
        include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_kerl.jl"))
        L = 1 + 16
        include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_kerl.jl"))
        L = 1 + 18
        include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_kerl.jl"))
        L = 1 + 20
        include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_kerl.jl"))
        L = 1 + 22
        include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_kerl.jl"))
        L = 1 + 24
        include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_kerl.jl"))
        L = 1 + 26
        include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_kerl.jl"))
        L = 1 + 28
        is_show_nuTi = true
        include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_kerl.jl"))
        
    end
end

