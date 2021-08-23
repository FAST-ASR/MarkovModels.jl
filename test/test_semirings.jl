# SPDX-License-Identifier: MIT

@testset "logaddexp" begin
    @test (@inferred MarkovModels.logaddexp(2.0, 3.0)) ≈ log(exp(2.0) + exp(3.0))
    @test (@inferred MarkovModels.logaddexp(10002.0, 10003.0)) ≈
        10000 + MarkovModels.logaddexp(2.0, 3.0)
end

