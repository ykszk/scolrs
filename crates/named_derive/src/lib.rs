use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput, Expr, ExprArray};

// #[macro_export]
// macro_rules! impl_named_w_lifetime_for {
//     ($name:ident, $draw_type:expr) => {
//         impl<'a> Named for $name<'a> {
//             fn name(&self) -> &'static str {
//                 stringify!($name)
//             }

//             fn draw_type(&self) -> &[&'static str] {
//                 $draw_type
//             }
//         }
//     };
// }
//
// impl_named_w_lifetime_for!(VertebralLabels, &[CLASS_ANNOTATION, CLASS_TEXT]);

#[proc_macro_derive(Named, attributes(draw_type))]
pub fn derive_named(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident;

    let draw_type = input
        .attrs
        .iter()
        .find(|attr| attr.path().is_ident("draw_type"))
        .map_or_else(
            || quote! { &[] },
            |attr| {
                let expr: Expr = attr
                    .parse_args()
                    .expect("draw_type attribute must be an array of const references");

                if let Expr::Array(ExprArray { elems, .. }) = expr {
                    let paths: Vec<_> = elems
                        .iter()
                        .map(|elem| {
                            if let Expr::Path(path_expr) = elem {
                                path_expr
                            } else {
                                panic!("Array elements must be const references")
                            }
                        })
                        .collect();

                    quote! { &[#(#paths),*] }
                } else {
                    panic!("draw_type must be an array")
                }
            },
        );

    let expanded = quote! {
        impl<'a> Named for #name<'a> {
            fn name(&self) -> &'static str {
                stringify!(#name)
            }

            fn draw_type(&self) -> &[&'static str] {
                #draw_type
            }
        }
    };

    TokenStream::from(expanded)
}
