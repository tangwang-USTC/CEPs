using JuMP
import HiGHS             # For large scale sparse linear optimization models.

model = Model(HiGHS.Optimizer)

set_optimizer_attribute(model, "presolve", "on")
set_optimizer_attribute(model, "time_limit", 60.0)

highs = Highs_create()

ret = Highs_setBoolOptionValue(highs, "log_to_console", false)

@assert ret == 0  # If ret != 0, something went wrong

Highs_addCol(highs, 1.0, 0.0, 4.0, 0, C_NULL, C_NULL)   # x is column 0

Highs_addCol(highs, 1.0, 1.0, Inf, 0, C_NULL, C_NULL)   # y is column 1

Highs_changeColIntegrality(highs, 1, kHighsVarTypeInteger)

Highs_changeObjectiveSense(highs, kHighsObjSenseMinimize)

senseP = Ref{Cint}(0)  # Instead of passing `&sense`, pass a Julia `Ref`

# Highs_getObjectiveSense(model, senseP)

senseP[] == kHighsObjSenseMinimize  # Dereference with senseP[]

Highs_addRow(highs, 5.0, 15.0, 2, Cint[0, 1], [1.0, 2.0])

Highs_addRow(highs, 6.0, Inf, 2, Cint[0, 1], [3.0, 2.0])

Highs_run(highs)

col_value = zeros(Cdouble, 2);

Highs_getSolution(highs, col_value, C_NULL, C_NULL, C_NULL)

Highs_destroy(highs)

col_value

