use proc_macro::TokenStream;
use quote::quote;
use syn::{
    parse_macro_input, Attribute, Data, DeriveInput, Expr, ExprArray, ExprLit, Lit, LitStr, Meta,
};

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

fn first_doc_line(attrs: &[Attribute]) -> proc_macro2::TokenStream {
    let doc_strings: Vec<String> = attrs
        .iter()
        .filter_map(|attr| match attr.meta {
            Meta::NameValue(ref name_value) if name_value.path.is_ident("doc") => {
                Some(&name_value.value)
            }
            _ => None,
        })
        .filter_map(|expr| match expr {
            Expr::Lit(ExprLit {
                lit: Lit::Str(s), ..
            }) => Some(s.value()),
            _ => None,
        })
        .collect();

    if doc_strings.is_empty() {
        quote! { None }
    } else {
        let first_line = doc_strings.first().unwrap().trim();
        quote! { Some(#first_line) }
    }
}

#[proc_macro_derive(Named, attributes(draw_type, label))]
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

    let label = input
        .attrs
        .iter()
        .find(|attr| attr.path().is_ident("label"))
        .map_or_else(
            || quote! { stringify!(#name) },
            |attr| {
                let lit: LitStr = attr
                    .parse_args()
                    .expect("label attribute must be a string literal");
                quote! { #lit }
            },
        );

    let first_doc_line = first_doc_line(&input.attrs);

    let expanded = quote! {
        impl<'a> Named for #name<'a> {
            fn id(&self) -> &'static str {
                stringify!(#name)
            }

            fn label(&self) -> &'static str {
                #label
            }

            fn description(&self) -> Option<&'static str> {
                #first_doc_line
            }

            fn draw_type(&self) -> &[&'static str] {
                #draw_type
            }
        }
    };

    TokenStream::from(expanded)
}

#[proc_macro_derive(ContentFilename)]
pub fn content_filename_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    let (content_field, filename_field, content_type) = match &input.data {
        Data::Struct(data_struct) => {
            let mut content_field = None;
            let mut filename_field = None;
            let mut content_type = None;

            for field in &data_struct.fields {
                if let Some(ident) = &field.ident {
                    if ident == "content" {
                        content_field = Some(ident);
                        content_type = Some(&field.ty);
                    } else if ident == "filename" {
                        filename_field = Some(ident);
                    }
                }
            }

            (content_field, filename_field, content_type)
        }
        _ => panic!("ContentFilename can only be derived for structs"),
    };

    let content_field = content_field.expect("Struct must have a field named 'content'");
    let filename_field = filename_field.expect("Struct must have a field named 'filename'");
    let content_type = content_type.expect("Struct must have a field named 'content'");

    let expanded = quote! {
        impl ContentFilename for #name {
            type ContentType = #content_type;

            fn content_filename(self) -> (Self::ContentType, String) {
                (self.#content_field, self.#filename_field)
            }

            fn new(content: Self::ContentType, filename: String) -> Self {
                Self {
                    #content_field: content,
                    #filename_field: filename,
                }
            }
        }
    };

    TokenStream::from(expanded)
}

/// Derive `TryFrom<&str>` for a struct that can be deserialized from JSON.
#[proc_macro_derive(TryFromJsonStr)]
pub fn try_from_json_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident;

    let expanded = quote! {
        impl TryFrom<&str> for #name {
            type Error = serde_json::Error;

            fn try_from(json: &str) -> Result<Self, Self::Error> {
                serde_json::from_str(json)
            }
        }
    };

    TokenStream::from(expanded)
}
