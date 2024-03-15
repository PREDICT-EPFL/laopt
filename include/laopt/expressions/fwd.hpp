#ifndef LAOPT_EXPRESSIONS_FWD_HPP
#define LAOPT_EXPRESSIONS_FWD_HPP

namespace laopt {

// forward declarations

template<typename Derived, typename Tag, typename Info, typename Capture>
class FunctionCapture;

template<typename Derived, typename EnableIf = void>
struct ExprEvaluator;

} // namespace laopt

#endif //LAOPT_EXPRESSIONS_FWD_HPP
