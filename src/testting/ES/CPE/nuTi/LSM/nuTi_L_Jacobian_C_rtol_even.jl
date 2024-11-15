
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
datatype = Float64
# is_change_datatype = true
is_change_datatype = false

rtol_OrjL = 1e-14
RDnuT = -1e-7

is_re_seed = true
is_re_seed = false
is_show_nuTi = false
is_optim =  true

is_norm_uhL = true
is_norm_uhL = false
        
    is_C = true 
    is_C = false 
        
    is_Jacobian = true
    # is_Jacobian = false

nModL = 3         # when `nMod` is larger, the optimization is more challenge! 
if nModL == -3
    datatype = BigFloat
    # is_C = 1, is_norm_uhL = 1, is_Jacobian = 0, |RDnuT| ≤ 1e-7: LM → 30
    # naiLok, uaiLok, vthiLok = zeros(datatype, nModL), zeros(datatype, nModL), zeros(datatype, nModL)
    naiLok = datatype.([0.3886465373727640857724535944144473751991655474589669512676112608912247647310806
              0.3791055204252116015363994591210717963998559758836975580125003640923826392385298
              0.232247942202024312691146946464480828400978476657335490719888375016392596030396])
    uaiLok = datatype.([-0.1263210461487311029831533050134379127922586193791323699118210517075793508041201
             0.133834998357290627157448524324238316211292351108801840654736434953339023004614
             0.2000000000000000000000000000000000000000000000000000000000000000000000000000004])
    vthiLok = datatype.([1.349450376542176742999017473535821525560206999645122430715001111302818427494118
               1.252233464458821633635706942390328219526258404106801329475552569693692044992016
               0.3983161589990016233652755840738502549135345962480762398094463190034895275138406])

    # naiL = deepcopy(naiLok)
    # uaiL = deepcopy(uaiLok)
    # vthiL = deepcopy(vthiLok)
end
njL = 3 * nModL

is_anasys_L = true
if is_anasys_L
    nL =  (28 - 0) / 2 + 1 |> Int
    errMhjL = zeros(njL,nL)
    errnuTM = zeros(nL,4)
    

    is_show_Dc = false
    if 1 == 1

        L = 0
        include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_Anasis_kerl.jl"))
        
        is_re_seed = false
        L = 2
        include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_Anasis_kerl.jl"))
        L = 4
        include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_Anasis_kerl.jl"))
        L = 6
        include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_Anasis_kerl.jl"))
        L = 8
        include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_Anasis_kerl.jl"))
        L = 10
        is_show_nuTi = true
        include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_Anasis_kerl.jl"))
        L = 12
        include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_Anasis_kerl.jl"))
        L = 14
        include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_Anasis_kerl.jl"))
        L = 16
        include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_Anasis_kerl.jl"))
        L = 18
        include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_Anasis_kerl.jl"))
        L = 20
        include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_Anasis_kerl.jl"))
        L = 22
        include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_Anasis_kerl.jl"))
        L = 24
        include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_Anasis_kerl.jl"))
        L = 26
        include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_Anasis_kerl.jl"))
        L = 28
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
        Lvec = 0:2:28
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

        L = 0
        include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_kerl.jl"))
        
        is_re_seed = false
        L = 2
        include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_kerl.jl"))
        L = 4
        include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_kerl.jl"))
        L = 6
        include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_kerl.jl"))
        L = 8
        include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_kerl.jl"))
        L = 10
        is_show_nuTi = true
        include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_kerl.jl"))
        L = 12
        include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_kerl.jl"))
        L = 14
        include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_kerl.jl"))
        L = 16
        include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_kerl.jl"))
        L = 18
        include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_kerl.jl"))
        L = 20
        include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_kerl.jl"))
        L = 22
        include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_kerl.jl"))
        L = 24
        include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_kerl.jl"))
        L = 26
        include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_kerl.jl"))
        L = 28
        is_show_nuTi = true
        include(joinpath(pathroot,"src/testting/ES/CPE/nuTi/LSM/nuTi_L_Jacobian_C_rtol_kerl.jl"))
        
    end
end

