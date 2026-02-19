module.exports = function (eleventyConfig) {
  eleventyConfig.addPassthroughCopy({ "public": "/" });

  const pathPrefix = process.env.PATH_PREFIX || "";
  eleventyConfig.addGlobalData("pathPrefix", pathPrefix);

  eleventyConfig.addFilter("topicTitle", function (topicId) {
    const topics = require("./src/_data/topics.json").topics;
    const t = topics.find((x) => x.id === topicId);
    return t ? t.title : topicId;
  });

  eleventyConfig.addFilter("relatedNotes", function (topicId, currentNoteId) {
    const topics = require("./src/_data/topics.json").topics;
    const t = topics.find((x) => x.id === topicId);
    if (!t || !t.notes) return { prev: null, next: null };
    const idx = t.notes.findIndex((n) => n.id === currentNoteId);
    if (idx < 0) return { prev: null, next: null };
    return {
      prev: idx > 0 ? t.notes[idx - 1] : null,
      next: idx < t.notes.length - 1 ? t.notes[idx + 1] : null,
    };
  });

  eleventyConfig.addCollection("notes", function (collectionApi) {
    return collectionApi.getFilteredByGlob("src/notes/**/*.md");
  });

  eleventyConfig.addCollection("topics", function (collectionApi) {
    const topics = require("./src/_data/topics.json").topics;
    return topics;
  });

  return {
    dir: {
      input: "src",
      output: "_site",
      includes: "_includes",
      data: "_data",
    },
    pathPrefix,
  };
};
