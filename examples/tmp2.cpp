    /*[[[cog nlp.begin_func("equal", ("a", "b")) ]]]*/
    //[[[end]]]
    // {
    //     out = a - b;
    // };
    /*[[[cog nlp.end_func("equal") ]]]*/
    //[[[end]]]

    // /*[[[cog nlp.gen_func("initial_state", ("x")) ]]]*/
    // //[[[end]]]
    // {
    //     out = x - x0;
    // };

    /*[[[cog nlp.gen_func("dynamics", ("xp", "x", "u")) ]]]*/
    //[[[end]]]
    // {
    //     out(0) = x0(1) * xp(0) - (-sin(x(1)) + x(1)*x(0)) + x0(0);
    //     out(1) = x0(0) * xp(1) - cos(x(0))*u(0) + x0(1);
    // };
    /*[[[cog nlp.gen_func("dynamics", ("xp", "x", "u")) ]]]*/
    //[[[end]]]
