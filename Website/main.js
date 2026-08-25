document.body.innerHTML = document.body.innerHTML.replaceAll('Avery Chen', 'Mark Dahl');

const filters = document.querySelectorAll('.filter');
const projects = document.querySelectorAll('.project');

filters.forEach((filter) => {
  filter.addEventListener('click', () => {
    filters.forEach((item) => item.classList.remove('active'));
    filter.classList.add('active');
    const selected = filter.dataset.filter;
    projects.forEach((project) => {
      const visible = selected === 'all' || project.dataset.category === selected;
      project.style.display = visible ? '' : 'none';
    });
  });
});
