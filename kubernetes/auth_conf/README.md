# About this two files

To enable Keycloak's tokens in Istio, new objects, `AuthorizationPolicy` and `RequestAuthentication`, are required. Without these objects, we would need to scrap cookies, as recommended by the KServe official documentation.

By implementing these objects, we can use OAuth tokens from Keycloak to make predictions. If these files do not exist, please deploy them manually.

We also added an `EnvoyFilter`, an Envoy extension for `Istio`, which enables us to manage responses within the Istio mesh and correctly return a `401 error` when the token is invalid.