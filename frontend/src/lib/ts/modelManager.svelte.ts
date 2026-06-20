export const modelState = $state({
  downloadedModels: ["Test Model", "Test Model #2"],
  chosenModel: "Test Model"
});

export function changeChosenModel(id: string) {
  modelState.chosenModel = id;
}