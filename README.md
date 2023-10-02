# ObjConsNLPModels.jl

![lifecycle](https://img.shields.io/badge/lifecycle-experimental-orange.svg)
[![build](https://github.com/tpapp/ObjConsNLPModels.jl/workflows/CI/badge.svg)](https://github.com/tpapp/ObjConsNLPModels.jl/actions?query=workflow%3ACI)
[![codecov.io](http://codecov.io/github/tpapp/ObjConsNLPModels.jl/coverage.svg?branch=master)](http://codecov.io/github/tpapp/ObjConsNLPModels.jl?branch=master)

<!-- Documentation -- uncomment or delete as needed -->
<!--
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://tpapp.github.io/ObjConsNLPModels.jl/stable)
[![Documentation](https://img.shields.io/badge/docs-master-blue.svg)](https://tpapp.github.io/ObjConsNLPModels.jl/dev)
-->

A Julia package that integrates into the [JuliaSmoothOptimizers](https://github.com/JuliaSmoothOptimizers/) ecosystem, allowing the user to define problems where it most efficient to calculate the objective and the constraint at the same time. Supports automatic differentation, and caches results to work with the JSO API.

The package is documented with docstrings, but it is fairly experimental. Suggestions and PRs welcome.
