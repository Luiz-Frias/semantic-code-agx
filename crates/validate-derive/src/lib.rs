//! Derive macro for `semantic_code_shared::Validate`.

use proc_macro::TokenStream;
use quote::{ToTokens, quote};
use syn::{
    Attribute, Data, DeriveInput, Fields, GenericArgument, Ident, Lit, Path, PathArguments, Type,
};

/// Derive `semantic_code_shared::Validate` with field-level checks.
#[proc_macro_derive(Validate, attributes(validate))]
pub fn derive_validate(input: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(input as DeriveInput);
    match expand_validate(&input) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

fn expand_validate(input: &DeriveInput) -> Result<proc_macro2::TokenStream, syn::Error> {
    let error_ty = parse_error_type(&input.attrs)?;
    let Data::Struct(struct_data) = &input.data else {
        return Err(syn::Error::new_spanned(
            input,
            "Validate can only be derived for structs",
        ));
    };

    let fields = match &struct_data.fields {
        Fields::Named(fields) => &fields.named,
        _ => {
            return Err(syn::Error::new_spanned(
                &struct_data.fields,
                "Validate requires named fields",
            ));
        },
    };

    let mut checks = Vec::new();
    for field in fields {
        let Some(ident) = field.ident.as_ref() else {
            continue;
        };
        let (field_name, validators) = parse_field_validators(&field.attrs, ident)?;
        if validators.is_empty() {
            continue;
        }
        let (is_option, inner_ty) = unwrap_option(&field.ty);
        for validator in validators {
            let check = match validator {
                Validator::NonEmpty => {
                    expand_non_empty(ident, &field_name, inner_ty, is_option, &error_ty)?
                },
                Validator::Range { min, max } => expand_range(
                    ident,
                    &field_name,
                    inner_ty,
                    is_option,
                    &error_ty,
                    &min,
                    &max,
                )?,
                Validator::Custom(path) => expand_custom(ident, &path, is_option),
            };
            checks.push(check);
        }
    }

    let name = &input.ident;
    Ok(quote! {
        impl semantic_code_shared::Validate for #name {
            type Error = #error_ty;

            fn validate(&self) -> Result<(), Self::Error> {
                #(#checks)*
                Ok(())
            }
        }
    })
}

fn parse_error_type(attrs: &[Attribute]) -> Result<Path, syn::Error> {
    let mut error_ty: Option<Path> = None;
    for attr in attrs {
        if !attr.path().is_ident("validate") {
            continue;
        }
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("error") {
                let value: syn::LitStr = meta.value()?.parse()?;
                let parsed: Path = value.parse()?;
                if error_ty.is_some() {
                    return Err(meta.error("duplicate validate(error = ...)"));
                }
                error_ty = Some(parsed);
                return Ok(());
            }
            Err(meta.error("unsupported validate attribute on container"))
        })?;
    }

    error_ty.ok_or_else(|| {
        syn::Error::new(
            proc_macro2::Span::call_site(),
            "missing #[validate(error = \"path\")] on struct",
        )
    })
}

fn parse_field_validators(
    attrs: &[Attribute],
    ident: &Ident,
) -> Result<(String, Vec<Validator>), syn::Error> {
    let mut validators = Vec::new();
    let mut field_name_override: Option<String> = None;
    for attr in attrs {
        if !attr.path().is_ident("validate") {
            continue;
        }
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("field") {
                let value: syn::LitStr = meta.value()?.parse()?;
                if field_name_override.is_some() {
                    return Err(meta.error("duplicate validate(field = ...)"));
                }
                field_name_override = Some(value.value());
                return Ok(());
            }
            if meta.path.is_ident("non_empty") {
                validators.push(Validator::NonEmpty);
                return Ok(());
            }
            if meta.path.is_ident("custom") {
                let value: syn::LitStr = meta.value()?.parse()?;
                let path: Path = value.parse()?;
                validators.push(Validator::Custom(path));
                return Ok(());
            }
            if meta.path.is_ident("range") {
                let mut min: Option<Lit> = None;
                let mut max: Option<Lit> = None;
                meta.parse_nested_meta(|nested| {
                    if nested.path.is_ident("min") {
                        let lit: Lit = nested.value()?.parse()?;
                        min = Some(lit);
                        return Ok(());
                    }
                    if nested.path.is_ident("max") {
                        let lit: Lit = nested.value()?.parse()?;
                        max = Some(lit);
                        return Ok(());
                    }
                    Err(nested.error("unsupported range attribute"))
                })?;
                let Some(min) = min else {
                    return Err(meta.error("range requires min"));
                };
                let Some(max) = max else {
                    return Err(meta.error("range requires max"));
                };
                validators.push(Validator::Range { min, max });
                return Ok(());
            }
            Err(meta.error("unsupported validate attribute on field"))
        })?;
    }
    let name = field_name_override.unwrap_or_else(|| ident.to_string());
    Ok((name, validators))
}

#[derive(Debug)]
enum Validator {
    NonEmpty,
    Range { min: Lit, max: Lit },
    Custom(Path),
}

fn unwrap_option(ty: &Type) -> (bool, &Type) {
    option_inner(ty).map_or((false, ty), |inner| (true, inner))
}

fn option_inner(ty: &Type) -> Option<&Type> {
    let Type::Path(type_path) = ty else {
        return None;
    };
    let segment = type_path.path.segments.last()?;
    if segment.ident != "Option" {
        return None;
    }
    let PathArguments::AngleBracketed(args) = &segment.arguments else {
        return None;
    };
    let mut type_arg = None;
    for arg in &args.args {
        if let GenericArgument::Type(inner) = arg {
            type_arg = Some(inner);
            break;
        }
    }
    type_arg
}

fn expand_non_empty(
    ident: &Ident,
    field_name: &str,
    ty: &Type,
    is_option: bool,
    error_ty: &Path,
) -> Result<proc_macro2::TokenStream, syn::Error> {
    if !is_string_like(ty) {
        return Err(syn::Error::new_spanned(
            ty,
            "non_empty can only be used on string-like fields",
        ));
    }
    let field_name = syn::LitStr::new(field_name, proc_macro2::Span::call_site());
    let err_expr = quote! {
        <#error_ty as semantic_code_shared::ValidationError>::empty(#field_name)
    };
    if is_option {
        Ok(quote! {
            if let Some(value) = self.#ident.as_ref() {
                if value.trim().is_empty() {
                    return Err(#err_expr);
                }
            }
        })
    } else {
        Ok(quote! {
            if self.#ident.trim().is_empty() {
                return Err(#err_expr);
            }
        })
    }
}

fn expand_range(
    ident: &Ident,
    field_name: &str,
    ty: &Type,
    is_option: bool,
    error_ty: &Path,
    min: &Lit,
    max: &Lit,
) -> Result<proc_macro2::TokenStream, syn::Error> {
    let kind = numeric_kind(ty)
        .ok_or_else(|| syn::Error::new_spanned(ty, "range can only be used on numeric fields"))?;
    let field_name = syn::LitStr::new(field_name, proc_macro2::Span::call_site());
    let min_str = syn::LitStr::new(&lit_to_string(min), proc_macro2::Span::call_site());
    let max_str = syn::LitStr::new(&lit_to_string(max), proc_macro2::Span::call_site());

    let finite_check = if matches!(kind, NumberKind::Float) {
        Some(quote! {
            if !value.is_finite() {
                return Err(<#error_ty as semantic_code_shared::ValidationError>::out_of_range(
                    #field_name,
                    value.to_string(),
                    #min_str.to_string(),
                    #max_str.to_string(),
                ));
            }
        })
    } else {
        None
    };

    if is_option {
        Ok(quote! {
            if let Some(value) = self.#ident.as_ref() {
                let value = *value;
                #finite_check
                if !(#min..=#max).contains(&value) {
                    return Err(<#error_ty as semantic_code_shared::ValidationError>::out_of_range(
                        #field_name,
                        value.to_string(),
                        #min_str.to_string(),
                        #max_str.to_string(),
                    ));
                }
            }
        })
    } else {
        Ok(quote! {
            let value = self.#ident;
            #finite_check
            if !(#min..=#max).contains(&value) {
                return Err(<#error_ty as semantic_code_shared::ValidationError>::out_of_range(
                    #field_name,
                    value.to_string(),
                    #min_str.to_string(),
                    #max_str.to_string(),
                ));
            }
        })
    }
}

fn expand_custom(ident: &Ident, path: &Path, is_option: bool) -> proc_macro2::TokenStream {
    if is_option {
        quote! {
            #path(self.#ident.as_ref())?;
        }
    } else {
        quote! {
            #path(&self.#ident)?;
        }
    }
}

fn is_string_like(ty: &Type) -> bool {
    match ty {
        Type::Reference(reference) => is_string_like(&reference.elem),
        Type::Path(type_path) => {
            let Some(segment) = type_path.path.segments.last() else {
                return false;
            };
            if segment.ident == "String" {
                return true;
            }
            if segment.ident == "str" {
                return true;
            }
            if segment.ident == "Box" {
                let PathArguments::AngleBracketed(args) = &segment.arguments else {
                    return false;
                };
                let mut inner = None;
                for arg in &args.args {
                    if let GenericArgument::Type(inner_ty) = arg {
                        inner = Some(inner_ty);
                        break;
                    }
                }
                if let Some(inner_ty) = inner {
                    return is_str_type(inner_ty);
                }
            }
            false
        },
        _ => false,
    }
}

fn is_str_type(ty: &Type) -> bool {
    match ty {
        Type::Path(type_path) => type_path
            .path
            .segments
            .last()
            .is_some_and(|segment| segment.ident == "str"),
        Type::Reference(reference) => is_str_type(&reference.elem),
        _ => false,
    }
}

#[derive(Copy, Clone, Debug)]
enum NumberKind {
    Integer,
    Float,
}

fn numeric_kind(ty: &Type) -> Option<NumberKind> {
    match ty {
        Type::Reference(reference) => numeric_kind(&reference.elem),
        Type::Path(type_path) => {
            let segment = type_path.path.segments.last()?;
            let ident = segment.ident.to_string();
            if matches!(ident.as_str(), "f32" | "f64") {
                return Some(NumberKind::Float);
            }
            if matches!(
                ident.as_str(),
                "u8" | "u16"
                    | "u32"
                    | "u64"
                    | "u128"
                    | "usize"
                    | "i8"
                    | "i16"
                    | "i32"
                    | "i64"
                    | "i128"
                    | "isize"
            ) {
                return Some(NumberKind::Integer);
            }
            None
        },
        _ => None,
    }
}

fn lit_to_string(lit: &Lit) -> String {
    match lit {
        Lit::Int(value) => value.base10_digits().to_string(),
        Lit::Float(value) => value.base10_digits().to_string(),
        Lit::Str(value) => value.value(),
        _ => lit.to_token_stream().to_string(),
    }
}
