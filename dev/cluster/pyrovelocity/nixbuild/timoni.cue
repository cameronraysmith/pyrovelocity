package main

import (
	templates "timoni.sh/nixbuild/templates"
)

values: templates.#Config

timoni: {
	apiVersion: "v1alpha1"

	instance: templates.#Instance & {
		config: values

		config: {
			metadata: {
				name:      string @tag(name)
				namespace: string @tag(namespace)
			}
			moduleVersion: string @tag(mv, var=moduleVersion)
			kubeVersion:   string @tag(kv, var=kubeVersion)
		}
	}

	apply: app: [for obj in instance.objects {obj}]

	if instance.config.test.enabled {
		apply: test: [for obj in instance.tests {obj}]
	}
}
